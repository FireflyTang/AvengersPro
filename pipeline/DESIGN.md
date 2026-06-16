# 设计方案:AvengersPro 三阶段路由流水线(embedding → 训练 → 在线推理)

## Context

原实现把所有逻辑堆在 `simple_cluster_router.py`(1231 行)和 `balance_cluster_router.py` 里,
通过命令行参数驱动,既做 embedding、又做聚类训练、又做评估,耦合严重、不便理解和复用。

目标场景:**只关心准确率,不关心成本**。训练数据是 `{query, 每个模型答对的概率}`。
embedding 使用用户自己的 OpenAI 兼容接口(服务端会自动截断)。

目标:拆成三个职责单一的脚本 + 一个共享配置模块,**参数硬编码在配置里、不走命令行**:
1. 生成 embedding(写入 SQLite 缓存)
2. Avengers 路由模型训练(聚类 + 每簇按准确率排名,保存模型产物)
3. 在线推理(可导入、线程安全的 Router 类 + JSONL 批处理调试入口)

已确定的简化:
- **去掉 balance/成本路由**,排名只按准确率(`records` 在簇内的均值)。
- **去掉 tiktoken 截断逻辑**,交给 embedding 服务端自动截断。
- **去掉 `MAX_ROUTER`/模型选择**:推理不再"选一个模型",而是输出该 query 下**每个模型的预测正确率**(完整 map,降序)。
- 核心算法:L2 归一化 → KMeans → 每簇存每个模型的平均准确率 → 推理时 top_k 最近簇 + softmax(beta) 按邻近度加权聚合各模型准确率。
- **预测正确率的定义**:`pred_acc[model] = Σ_(top_k 簇) prob × 该簇内该模型的平均准确率`。
  即用簇内真实准确率(存的 `scores` 字段)做邻近度加权平均,得到 [0,1] 的可解释"正确率",
  **而非**原算法用于选择的 `1/(rank+1)` 排名分。

## 目录结构

```
pipeline/
  config.py              # 共享常量(唯一改参数的地方)
  embedding_cache.py     # 自带副本:SQLite 缓存 + 指数退避重试 + 原始请求/返回留存
  _common.py             # 共享逻辑:数据加载、按 dataset 切分、预测聚合数学
  generate_embeddings.py # 脚本1
  train.py               # 脚本2
  inference.py           # 脚本3:Router 类 + __main__ 批处理
```
- **`pipeline/` 完全自包含,不 import 仓库根目录任何遗留文件**。
- `pipeline/embedding_cache.py` 是根目录 `embedding_cache.py` 的副本,在其基础上扩展:
  - 构造参数 `max_retries` / `initial_delay` 由 config 透传(重试次数可配,含指数退避)。
  - 新增可选 `raw_log_path`:每次**真实 API 调用**(缓存未命中)时,把原始请求 + 原始返回追加写入 JSONL 审计文件(加锁并发安全)。
- 现有 `simple_cluster_router.py` / `balance_cluster_router.py` / `config.py` / `embedding_cache.py` **保持不动**,与 pipeline 互不影响。

## pipeline/config.py(共享常量)

```python
# --- Embedding 服务(用户填,OpenAI 兼容,服务端自动截断) ---
EMBEDDING_BASE_URL = "http://localhost:8000/v1"
EMBEDDING_API_KEY  = "dummy-key"
EMBEDDING_MODEL    = "your-embedding-model"
EMBEDDING_MAX_WORKERS = 4                  # 并发数(驱动 ThreadPoolExecutor)
EMBEDDING_MAX_RETRIES = 5                  # 报错自动重试次数
EMBEDDING_RETRY_INITIAL_DELAY = 1.0        # 重试初始退避秒数(指数退避 ×2)
EMBEDDING_RAW_LOG_PATH = ".cache/embedding_raw_log.jsonl"  # 原始请求/返回留存(None=关闭)
CACHE_DIR = ".cache"                        # SQLite 缓存目录

# --- 数据路径 ---
TRAIN_INPUT_PATH  = "data/train.jsonl"      # {query, records:{model:acc}, dataset?, index?}
INFER_INPUT_PATH  = "data/queries.jsonl"    # {query}(批处理调试用)
INFER_OUTPUT_PATH = "results/routing.jsonl"
MODEL_DIR         = "models/router"         # 训练产物输出 / 推理加载目录

# --- 聚类 & 路由参数 ---
N_CLUSTERS  = 32
SEED        = 42
TRAIN_RATIO = 0.7      # <1.0 → 切分并评估; >=1.0 → 全量训练、不评估
TOP_K       = 3        # 推理时聚合最近的几个簇
BETA        = 9.0      # 簇距离 softmax 温度(top_k=1 时无影响)
EXCLUDED_MODELS = []   # 可选:从排名中剔除的模型
```
(无 `MAX_ROUTER`——推理输出全部模型的预测正确率,不做选择。)

## 脚本1:generate_embeddings.py

读 `TRAIN_INPUT_PATH`(并入 `INFER_INPUT_PATH`),取出所有 query,并发调用 `EmbeddingCache.get(query)`
把向量写入 SQLite 缓存(命中即跳过)。**不做截断**。
- **并发数**由 `EMBEDDING_MAX_WORKERS` 控制(`ThreadPoolExecutor`)。
- **重试**:`EmbeddingCache(max_retries=..., initial_delay=...)` 内置指数退避。
- **原始留存**:`raw_log_path=EMBEDDING_RAW_LOG_PATH`;每次真实调用追加一行 JSONL:
  `{timestamp, model, request:{input, model}, response:<完整原始返回>}`,加锁保证并发写安全。
  ⚠️ 原始返回含完整向量,文件会较大;设 `None` 可关闭。
- 缓存 key = `md5(原始 query) + EMBEDDING_MODEL`(与脚本2/3 一致)。
- 输出:打印总数 / 新增 / 命中数;产物:`.cache/embeddings.db` + `embedding_raw_log.jsonl`。

## 脚本2:train.py

用缓存里的 embedding 训练路由模型并保存产物。
1. 读 `TRAIN_INPUT_PATH`,校验 `query`/`records`,`records` 值转 float(bool/None/数值)。`usages` 忽略。
2. `available_models` = 第一条 records 的键 - `EXCLUDED_MODELS`。
3. 对每条 query 调 `embedder.get()`(全命中缓存),L2 `Normalizer().fit_transform`。
4. **每簇准确率表**:簇内对每个模型求 `records` 均值,存 `{scores:{model:mean_acc}, ranking:[按 acc 降序的模型]}`。
5. **切分逻辑**:
   - `TRAIN_RATIO < 1.0`:按 dataset 分组后切分;train 上 fit KMeans + 建簇准确率表;test 上评估。
   - `TRAIN_RATIO >= 1.0`:全量 fit,跳过评估。
6. **评估(反映质量)**:对每条 test query 算 `pred_acc[model]`,取 argmax 模型作为"路由选择",
   用它的真实 `records` 值累计 → 打印:整体准确率(各 dataset 简单平均)、每 dataset 准确率、
   路由选择分布、预测 MAE(预测 vs 真实)、平均 regret(最优 − 路由模型)。
7. **保存产物**到 `MODEL_DIR`:`normalizer.joblib`、`cluster_centers.npy`、
   `cluster_rankings.json`(簇 id 存为字符串)、`metadata.json`(`available_models`/`embedding_model`/`n_clusters`/`top_k`/`beta`)。
8. **训练期中间打印**:每簇样本数分布、每簇 top 模型及其平均准确率、各模型整体平均准确率。

## 脚本3:inference.py

`class Router`(线程安全)+ `__main__` 批处理。
- `__init__(model_dir=MODEL_DIR)`:加载 `normalizer.joblib`、`cluster_centers.npy`、
  `cluster_rankings.json`(**字符串簇 id → int**)、`metadata.json`;建 `EmbeddingCache`。加载后状态只读。
- `predict(query) -> dict[str, float]`:返回**每个模型的预测正确率**,降序。
  embed → `normalizer.transform` → `dist = 1 - q @ centers.T` → 取 `TOP_K` 最近簇 →
  `prob = softmax(-BETA * dist)` → `pred_acc[model] = Σ prob × 簇内该模型平均准确率`。
- `predict_batch(queries)`:向量化批处理。
- **线程安全**:`predict` 只读已加载的 numpy 数组、不改 `self`;numpy 运算无状态;
  `EmbeddingCache` 内部 OpenAI client 线程安全、SQLite 用 `check_same_thread=False`+WAL+每次操作独立连接。
- `__main__`:读 `INFER_INPUT_PATH` → `predict_batch` → 写 `INFER_OUTPUT_PATH`
  (每行 `{query, predicted_accuracy:{model:acc}}`,**输出全部模型分数,不标注最优模型**),并打印调试信息。

## 移植来源(逻辑参照,不产生 import 依赖)

| 逻辑 | 移植自 |
|---|---|
| `EmbeddingCache`(SQLite 缓存 + 指数退避重试,新增 raw_log) | `embedding_cache.py`(整体拷入后扩展) |
| 并发 embedding 生成框架 | `simple_cluster_router._generate_embeddings_concurrent`(去截断) |
| records 值规范化(bool/None/数值→float) | `_validate_data_item`(simple_cluster_router.py:104-145) |
| 按 dataset 分组 + 切分 | `load_and_split_data`(simple_cluster_router.py:255-294) |
| 每簇准确率表(均值 scores) | `_compute_cluster_rankings`(simple_cluster_router.py:409-453) |
| 距离/softmax 聚合(top_k/beta;聚合目标改为真实准确率) | `route_queries_batch`(simple_cluster_router.py:527-567) |
| 产物保存格式 | `export_cluster_models`(simple_cluster_router.py:455-518) |

## 注意事项 / 已知坑

- `cluster_rankings.json` 簇 id 是字符串,推理加载时必须 `int()` 回去,否则匹配不上 KMeans 的 int label。
- 必须用**训练时保存的同一个 normalizer** 处理推理 query,不能临时新建。
- `TRAIN_RATIO` 切分时必须严格 `0<ratio<1`;`>=1.0` 单独分支(全量、不评估)。
- 推理 query 的缓存 key 与训练一致(都不截断),训练集出现过的 query 推理时会命中缓存。

## 验证(需配置好 embedding endpoint)

1. 准备 `data/train.jsonl`(模型名 + records 0~1 概率 + `dataset` 字段)。
2. 设 `pipeline/config.py` 的 embedding 三项 + `TRAIN_RATIO`。
3. `python pipeline/generate_embeddings.py` → 打印 embedded/命中数,生成 `.cache/embeddings.db` 与 raw log。
4. `python pipeline/train.py` → 打印簇分布/准确率/MAE/regret;`models/router/` 下出现四个产物文件。
5. `python pipeline/inference.py` → 控制台打印每条 query 各模型预测正确率;`results/routing.jsonl` 写出全部分数。
6. 程序化:`from pipeline.inference import Router; r=Router(); r.predict("...")`;多线程并发调 `predict` 验证线程安全。

> 本方案已实现并通过端到端验证(合成数据 + 预填缓存):训练/评估/推理/线程安全均通过。
