# 实验模块重构计划（更新版）

## 模块范围（已澄清）
本模块不是单独的绘图，而是**完整实验流程**，包含三部分：
- 实验生成：`scientific_research_work/experimenting.py`
- 绘图生成：`scientific_research_work/drawing.py`
- 实验写作：`scientific_research_work/experiment_writing.py`

## 目标（对齐整体重构方向）
- 入口函数固定为 `experiment()`（或 `experiment_pipeline()`），由“调度中心”统一调用
- 移除脚本式 CLI / `sys.path` hack，全部配置驱动
- 将实验生成/绘图/写作拆成清晰子模块并可独立调用
- **LLM 调用统一使用 `D:\experiment_auto_write_old\llm`**
  - 优先复用 `LLMClient`、`llm.message`、`LLM_PRESETS`
  - 如遇不满足需求的场景，尽量通过**补充 preset**解决（最小新增）
- `config/config.yaml + problems/*.yaml` 统一控制参数

## 现状问题（需处理）
- 三个脚本混在根目录，职责耦合、路径硬编码、CLI 强依赖
- 大量 `agent` 类直连，调用方式不统一
- `sys.exit(1)`/调试残留阻断流程
- 复用函数（rerank、encode_image、text_image_separate）散落多处
- config 字段分散在脚本内，不利于跨模块调度

## 新目录布局（草案）
```
scientific_research_work/
  experiment/
    __init__.py              # 暴露 experiment()，内部可选 mode: get/doing/draw/write/all
    settings.py              # 读取 config + problems，输出统一实验配置

    pipeline/
      orchestrator.py        # experiment() 入口（串联分析视角/实验生成/绘图/写作）
      analysis_perspective.py# 分析视角生成（从 experimenting.py 中拆出）
      experiment_generation.py# 实验生成与判别（从 experimenting.py 中拆出）
      drawing.py             # 绘图生成（原 drawing.py）
      writing.py             # 实验写作（原 experiment_writing.py）
      prepare.py             # 目录准备/清理

    retrieval/
      vectorstore.py         # Chroma 加载/查询
      rerank.py              # reranking_intercept()

    prompts/
      __init__.py
      experimenting.py       # EXPERIMENT_ANALYSIS_SYSTEM / MODULE_DESCRIPTION_SYSTEM 等
      drawing.py             # DRAW_CHART_SYSTEM / DRAW_TABLE_SYSTEM 等
      writing.py             # EXPERIMENT_WRITE_* / TYPESETTING / GENERATE_TEX 等

    utils.py                 # clear_folder / encode_image / text_image_separate 等

  common/
    logging.py               # 进程/脚本通用日志初始化（setup_logging）
    image.py                 # encode_image（供 database_building / experiment / writing 复用）
    fs.py                    # clear_folder / ensure_dir 等通用文件操作
    config.py                # _deep_merge / _resolve_path 等通用配置工具（可选）

  llm/
    embeddings.py            # get_Embedding_model / call_Embedding_model / SiliconFlowEmbeddings
    rerank.py                # get_reranker_result / reranking_intercept
    tokens.py                # num_tokens_from_string 统一入口（可选）
```

## LLM 调用约束（必须遵守）
- 全部 LLM 调用通过 `llm.LLMClient.call()` 完成
- 消息构造使用 `llm.message.system / user_text / user_image_base64 / user_content`
- `agent` 类不再保留，改为 **preset + prompt** 组合
- `llm/presets.py` 仅新增必要的 preset（尽量少）
  - 示例（待确认）：
    - `EXPERIMENT_ANALYSIS` / `ANALYSIS_PERSPECTIVE`
    - `EXPERIMENT_INDICATORS` / `INFORMATION_ARCHITECTURE`
    - `WRITE_OUTLINE` / `WRITE_TASK` / `THINK_TASK`
    - `INTEGRATION_WRITE` / `GENERATE_TEX` / `TYPESETTING`
    - `DRAWING_ANALYSIS` / `DRAWING_CODE` / `DRAWING_CODE_CHECK`

## utils.py 拆分与环境变量（新增要求）
- **敏感信息移入 `.env`**（根目录）：  
  - `OPENAI_API_KEY`  
  - `OPENAI_BASE_URL`  
  - `RERANK_URL` / `RETRIEVE_URL`  
  - `RERANK_RETRIEVE_KEY`  
  - `PDF_MD_TOKEN`  
  - 新增 `.env.example` 供配置参考
- **LLM 相关迁移到 `llm/`**：  
  - `get_client()` 移到 `llm/client.py`（或 `llm/factory.py`）  
  - `input_logger` / `output_logger` / `num_tokens_from_string` 移到 `llm/logging.py`  
  - `llm/client.py` 只从 `llm/logging.py` 读取 token 统计与日志器
- **检索/嵌入相关迁移到 `llm/`**：  
  - `SiliconFlowEmbeddings`、`call_Embedding_model()` → `llm/embeddings.py`  
  - `get_reranker_result()` / `reranking_intercept()` → `llm/rerank.py`
- **通用日志函数迁移**：  
  - `setup_logging()` → `scientific_research_work/common/logging.py`
- **utils.py 保留为薄入口（或清空）**：  
  - 仅做向后兼容的 re-export（可选），逐步移除引用

## database_building 内部 utils 依赖调整（新增要求）
需同步替换 `scientific_research_work/database_building` 对 `utils.py` 的依赖：
- `pipeline/pdf_extract.py`：`PDF_MD_TOKEN` → 从 `.env` 或 `config.yaml` 读取  
  - 推荐：`database_building/settings.py` 提供 `pdf_md_token` 字段
- `text/split.py`：`num_tokens_from_string` → 移到 `llm/logging.py` 或 `llm/tokens.py`  
  - 由 `llm` 统一提供 token 计数工具
- `vector_db/build.py`：  
  - `call_Embedding_model` / `get_Embedding_model` → `llm/embeddings.py`  
  - `setup_logging` → `scientific_research_work/common/logging.py`
- `README.md` 中提到的 `utils.get_client()` → 改为 `llm.LLMClient` 或新的 `llm.client.get_client`

## 配置设计（config/）
### config.yaml 新增（建议）
```
experiment:
  mode: get | doing | draw | write | all
  pool_size: 4
  devices: ["no_device", 2]
  embedding_model: BAAI/bge-m3
  embedding_model_source: api
  reranker_path: BAAI/bge-reranker-v2-m3
  reranker_model_source: api
  database_device: cpu
  text_model_name: gpt-4.1-mini
  multimodal_model_name: gpt-4.1-mini
  judge_model_name: gpt-4.1
  temperature: 1.0
  retrieval_data_num: 20
  reranked_data_num: 8
  analysis_perspective_num: 1

drawing:
  retrieval_template_num: 5
  max_try: 5

experiment_writing:
  integration_model_name: gpt-4.1-mini
  temperature: 1.0
```

### problems/*.yaml 新增（建议）
- `problem_id`
- `bib_path`
- `experiment_code_path`
- `method_code_path`
- `experimental_data_explain_path`
- `tex_template_path`
- `result_root`（或默认按 `paths.result_dir` 拼装）
- （可选）`perspectives`（指定 analysis_perspective_XXX）

`settings.py` 负责把 `config.yaml` 与 `problems/*.yaml` 合并成统一配置，路径尽量由 `paths.*` 推导，避免硬编码。

## 旧代码到新模块映射（建议）
- `experimenting.py`
  - `get_method_exlain` → `pipeline/analysis_perspective.py`
  - `get_experimental_indicators` → `pipeline/analysis_perspective.py`
  - `mode_select == 'get'` → `pipeline/analysis_perspective.py`
  - `doing_experiment` / `mode_select == 'doing'` → `pipeline/experiment_generation.py`
- `drawing.py`
  - `draw_action` → `pipeline/drawing.py`
  - `drawing` → `pipeline/drawing.py`
- `experiment_writing.py`
  - `TaskMemory` / `execute_task` / `integration_writing` → `pipeline/writing.py`
- `reranking_intercept` → `llm/rerank.py`
- `encode_image` → `common/image.py`
- `text_image_separate` → `experiment/utils.py`（或 `common/multimodal.py`）
- `clear_folder` → `common/fs.py`

## database_building 公用函数抽取（新增要求）
从 `scientific_research_work/database_building` 抽可复用函数到 `common/` 或 `llm/`：
- `pipeline/md_postprocess.py`：`encode_image()` → `common/image.py`  
  - 原文件保留薄封装或直接改 import
- `core.py`：`_clear_folder()` → `common/fs.py`  
  - `_init_logging()` 统一改用 `common/logging.py`
- `settings.py`：`_deep_merge()` / `_resolve_path()` → `common/config.py`（可选）  
  - 便于其它模块复用同样的 config 逻辑
- `text/split.py`：`num_tokens_from_string` → `llm/tokens.py`  
  - 统一 token 统计入口

## 关键流程（模块内）
1. **experimenting(get)**：生成分析视角 + 可视化策略
2. **experimenting(doing)**：按视角生成实验代码 + 数据，并用 judge 校验
3. **drawing**：检索绘图模板 → 生成绘图代码 → 校验与重试
4. **experiment_writing**：写作计划 → 分步写作 → LaTeX 生成与排版
5. **orchestrator**：根据 `mode` 串联上述步骤

## 具体步骤
1. 新建 `scientific_research_work/experiment/` 包与骨架
2. 迁移 prompts（只迁本模块使用的）到 `experiment/prompts/`
3. 用 `llm` 统一调用替换所有 `agent` 类
4. 拆分 `experimenting/drawing/experiment_writing` 逻辑到 pipeline
5. 统一 rerank / encode / text-image 分离等公共函数
6. 移除 CLI/硬编码配置，入口函数改为配置驱动
7. 清理未使用 import、调试残留与重复逻辑

## 最小验证
- 获取分析视角：
  - `python -c "from scientific_research_work.experiment import experiment; from scientific_research_work.common.settings import load_experiment_config; experiment(load_experiment_config(), mode='get')"`
- 执行实验：
  - `python -c "from scientific_research_work.experiment import experiment; from scientific_research_work.common.settings import load_experiment_config; experiment(load_experiment_config(), mode='doing')"`
- 生成绘图：
  - `python -c "from scientific_research_work.experiment import experiment; from scientific_research_work.common.settings import load_experiment_config; experiment(load_experiment_config(), mode='draw')"`
- 实验写作：
  - `python -c "from scientific_research_work.experiment import experiment; from scientific_research_work.common.settings import load_experiment_config; experiment(load_experiment_config(), mode='write')"`

---
如需我继续，我可以按该计划开始拆分与迁移。
