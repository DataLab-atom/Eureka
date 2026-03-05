# database_building 模块重构计划（更新版）

## 目标（已确认）
- 入口函数固定为 `database_building()`
- 移除 CLI 与 `vb_mode` 分支，全部改为配置驱动
- 根目录新增 `config/`，用 `config.yaml + problems/` 机制切换问题
- 本模块所用的智能体与提示词全部迁入并整理
- 建立统一的 LLM 调用接口，供后续模块复用
- 现有 agent（如 ChartAgent、CheckReferenceRepeatAgent）不再保留独立“Agent 类”，统一沉到 LLM 调用模块

## 新目录布局（草案）
```
config/
  config.yaml                # 全局/模块超参数 + 当前问题切换
  problems/
    demo.yaml                # 测试问题

scientific_research_work/
  database_building/
    __init__.py              # 暴露 database_building()
    settings.py              # 配置 schema / 默认值 / 合并逻辑
    pipeline/
      pdf_extract.py         # pdf_extractor / load_extractor_pdf / extractor_zip / do_parse / _process_output
      md_postprocess.py      # data_dealing / chart_squeezing / encode_image
    vector_db/
      build.py               # build_paper_vb / build_plotcode_vb 等向量库构建
      stats.py               # count_vector_database
    text/
      split.py               # split_text_with_headings
    scholar/
      semantic.py            # SemanticScholar 相关搜索与引用/被引扩展逻辑
    prompts/
      __init__.py
      database_building.py   # 本模块相关 prompts / output schema
    utils.py                 # 模块内轻量通用工具

llm/
  __init__.py
  client.py                  # 统一 LLM 调用接口
  message.py                 # 统一消息构造（文本/图片）
  types.py                   # Request/Response/Schema 类型定义
  presets.py                 # 统一“智能体名称=调用配置”的属性表
```

## 配置设计（config/）
- `config.yaml` 包含：
  - `active_problem`（字符串，用于切换 `problems/*.yaml`）
  - `paths`（root、数据路径、输出路径等）
  - `llm`（模型名、温度、重试策略、provider）
  - `database_building` 超参数（embedding、pdf 解析方式、chunk 配置等）
- `problems/demo.yaml`：提供最小可运行的测试问题与必要参数
- `settings.py` 负责：加载 `config.yaml` → 合并问题 YAML → 输出统一配置 dict

## 统一 LLM 接口（llm/）
- 目标：替代当前 `OpenaiClient / DeepseekClient / OpenaiJsonClient` + 多种 Agent 类的混乱用法
- 设计要点：
  - `LLMClient.call()` 统一请求入口
  - `LLMMessage` 统一文本/图片消息构造
  - `response_schema`（pydantic）可选，统一结构化输出解析
  - 统一重试、日志、token 统计

### “智能体”收敛方式（关键变更）
- 不再保留 `ChartAgent / CheckReferenceRepeatAgent / ...` 等独立类
- 在 `llm/presets.py` 中用 **属性名** 统一登记：
  - 如 `LLM_PRESETS.CHART_DESC`、`LLM_PRESETS.CHECK_REFERENCE_REPEAT`
  - 每个属性包含：`model`、`temperature`、`prompt_template`、`output_schema`
- “嵌套死的提示词模板”直接写到调用处，避免额外包装层

## 智能体与提示词迁移（本模块范围）
- 从 `agent/` 与 `prompt/` 中抽取 **本模块实际使用** 的提示词与 schema
- 迁入 `scientific_research_work/database_building/prompts`
- 在 `llm/presets.py` 中登记对应的 preset 名称
- 业务代码中直接调用：
  - `llm.call(preset=LLM_PRESETS.X, messages=[...])`

## 具体步骤
1. 建立 `config/` 目录与 `config.yaml`、`problems/demo.yaml`
2. 新增 `llm/` 统一接口与 `presets.py`
3. 搭建 `database_building/` 子包骨架
4. 迁移 prompts / schema；删掉独立 Agent 类改为 presets
5. 迁移与拆分原脚本函数到对应子模块
6. 移除 CLI 与 `vb_mode` 分支，入口改为 `database_building()`
7. 清理编码/重复 import/未使用依赖，做最小验证

## 最小验证
- 使用 `config.yaml + problems/demo.yaml` 跑一次 `database_building()`
- 验证：PDF 解析 → Markdown 处理 → 向量库构建 → SemanticScholar 连接
