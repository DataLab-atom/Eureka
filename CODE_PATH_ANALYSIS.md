# Eureka 代码路径全流程分析

## 一、项目概述

**Eureka** 是一个端到端的自动化科研论文生成系统。给定一个研究问题、参考代码和种子参考文献，它能自动爬取文献、设计并运行实验、生成完整的带引用的 LaTeX 科研论文。

## 二、总体流水线（Pipeline）

流水线定义在 `scientific_research_work/pipeline.py`，共 **6 个阶段**，按顺序执行：

```
database_building → experiment → method_writing → related_work_writing → introduction_writing → merge_writing
```

入口是 `main.py`，通过命令行参数可选择执行哪些步骤：
```bash
python main.py --steps database_building,experiment,method_writing,...
```

## 三、各阶段详细分析

### 阶段 1：`database_building`（文献数据库构建）
**核心文件**: `scientific_research_work/database_building/core.py`

**流程：**
1. **构建绘图代码向量数据库** — 将 `plot_code_template/` 中的绘图模板建成向量索引（ChromaDB），供后续实验绘图时检索
2. **爬取种子文献** — 从 `reference.bib` 出发，通过 `paper_crawler` 下载 PDF 论文
3. **PDF 转 Markdown** — 用 `pdf_extractor` 将 PDF 转为 Markdown 文本
4. **构建实验论文向量数据库** — 将文本切块后嵌入（BAAI/bge-m3），存入 ChromaDB
5. **引文网络扩展** — 通过 Semantic Scholar API，沿着种子论文的**引用**和**被引**链条递归爬取更多论文，直到达到 `all_paper_num` 上限
6. **查重去重** — 用 `check_reference_repeat` 避免重复下载
7. **构建全文向量数据库** — 将所有新爬到的论文也转为 Markdown 并入库

**输出：** 两个 ChromaDB 向量数据库（实验用 + 全文用）、论文信息 JSON、PDF 和 Markdown 文件

---

### 阶段 2：`experiment`（实验设计与执行）
**核心文件**: `scientific_research_work/experiment/pipeline/orchestrator.py`

这一阶段分为 4 个子步骤，由 `run_all()` 串联：

#### 2.1 分析视角生成 (`analysis_perspective.py`)
- 读取用户提供的方法代码（`Ours.py`）和基线方法代码
- LLM 提取论文摘要、主要研究方向、方法命名
- 从向量数据库检索相关背景知识（文本+图像）
- LLM 生成多个**分析视角**（analysis_perspective），每个包含：
  - 分析角度、核心关注点、差异化特征、预期洞见、实验指标
  - 可视化策略（适合什么类型的图表）

#### 2.2 实验执行 (`experiment_generation.py`)
- 根据每个分析视角，LLM 生成 Python 实验脚本
- 支持多设备并行执行（CPU/GPU）

#### 2.3 绘图 (`drawing.py`)
- 检索绘图模板向量数据库，找到匹配的绘图代码模板
- LLM 生成绘图脚本，执行产生实验结果图表

#### 2.4 实验写作 (`writing.py`)
- LLM 根据实验结果图表写出实验章节的 LaTeX 文本

**输出：** 实验代码、结果图表、`experiment.tex`

---

### 阶段 3：`method_writing`（方法论写作）
**核心文件**: `scientific_research_work/method_writing/core.py`

**流程：**
1. **构建写作需求** — 从方法代码和研究信息中生成写作要求
2. **生成写作大纲** — LLM 以 JSON 格式输出层级化的写作计划，包含 `think` 和 `write` 两种任务类型
3. **递归执行任务树** — `TaskMemory` 管理任务状态：
   - `think` 任务：先思考后产出思路
   - `write` 任务：基于思路产出文本
   - 子任务间的思考结果作为上下文传递
4. **整合写作** — 将所有碎片文本整合为连贯段落
5. **迭代优化** — 多轮 `improve_round` 自我改进
6. **LaTeX 转换** — 先生成 LaTeX 代码，再排版优化

**输出：** `method.tex`

---

### 阶段 4：`related_work_writing`（相关工作写作）
**核心文件**: `scientific_research_work/related_work_writing/core.py`

**流程：**
1. **划分研究领域** — LLM 将研究内容分为 `field_num` 个相关领域
2. **逐领域、逐句写作** — 对每个领域，每次写 `sentence_num` 句：
   - 先**思考**下一句应写什么
   - 从全文向量数据库 **RAG 检索**相关文献（含 reranking）
   - 选择引用（去重，控制数量）
   - LLM 结合参考文献写出带引用的 LaTeX 文本
3. **Judge 验证** — 用 `MultimodalJudgeAgent` 检查是否引用了不存在的文献，最多重试 3 次
4. **整合** — 将各领域内容合并为完整的 Related Work 章节
5. **引用清理** — 移除未在正文中实际使用的 bib 条目

**输出：** `related_work.tex`、`reference_bib.bib`

---

### 阶段 5：`introduction_writing`（引言写作）
**核心文件**: `scientific_research_work/introduction_writing/core.py`

**流程：**
1. **读取已有章节** — 将已完成的 related_work、method、experiment 章节作为上下文
2. **生成写作大纲** — 与 method_writing 类似的 `TaskMemory` 机制，但每个任务有 `need_cite` 标记
3. **带引用的任务执行**：
   - 对需要引用的任务，先 `think`，再 RAG 检索，检查引文去重后写作
   - **虚构引用检测** — `call_check_reference_fiction` 验证 LLM 是否捏造了参考文献
   - 不需要引用的任务直接写作
4. **整合与引用清理** — 整合段落，清理未使用的 bib

**输出：** `introduction.tex`、追加到 `reference_bib.bib`

---

### 阶段 6：`merge_writing`（论文合并与最终生成）
**核心文件**: `scientific_research_work/merge_writing/core.py`

**流程：**
1. **收集所有章节** — 读取 introduction、related_work、method、experiment 四个 `.tex` 文件
2. **生成 Conclusion、Abstract、Title** — LLM 依次生成结论（基于全文）、摘要（基于全文+结论）、标题（基于全部内容）
3. **初始合并** — 用 LaTeX 模板将所有内容合并为完整论文
4. **迭代优化** — `improve_round` 轮自我改进
5. **提交检查** — `check_round` 轮格式检查
6. **进化优化** — `evolve_paper()` 使用进化算法优化论文：
   - 将论文拆分为 4 个 section（Introduction, Related Work, Method, Experiment）
   - `TextModifierAgent` 修改每个 section
   - `TextEvalAgent` 评判质量
   - 多代种群演化，择优保留
7. **引用清理** — 最终确认 bib 中只保留实际使用的条目
8. **合并 bib 来源** — 去重合并所有阶段产生的 bib

**输出：** `full_paper.tex`、`reference_bib.bib`（最终版）

---

## 四、LLM 调用层
**核心文件**: `llm/client.py`

- `LLMClient` 封装了 OpenAI 兼容 API 的调用
- 支持三种模式：
  - **普通文本** — `chat.completions.create`
  - **JSON 模式** — `response_format: json_object`
  - **结构化输出** — `beta.chat.completions.parse` + Pydantic schema
- 内置重试（最多 3 次）和 token 计数日志
- 支持多模态输入（文本 + 图像 base64）

## 五、关键支撑组件

| 组件 | 路径 | 功能 |
|------|------|------|
| 向量数据库 | `database_building/vector_db/` | ChromaDB 构建（论文 + 绘图模板） |
| Embedding | `llm/embeddings.py` | BAAI/bge-m3 嵌入（本地/API） |
| Reranking | `llm/rerank.py` | bge-reranker-v2-m3 重排序 |
| 论文爬虫 | `common/paper_crawler/` | Semantic Scholar + arXiv + Unpaywall |
| PDF 解析 | `database_building/pipeline/pdf_extract.py` | PDF → Markdown |
| Judge 系统 | `common/judge_evol/` | 多模态评判 Agent + 进化优化 |
| 配置系统 | `common/settings.py` | YAML 配置加载与路径解析 |

## 六、数据流总结

```
输入:
  ├── reference.bib              (种子参考文献)
  ├── code/Ours.py               (用户的方法代码)
  ├── code/other/*/              (基线方法代码)
  ├── experimental_environment/  (实验数据集)
  └── template/template.tex      (LaTeX 模板)

处理:
  database_building: bib → 爬取PDF → 转Markdown → 向量数据库
  experiment:        代码 + 向量DB → 分析视角 → 实验脚本 → 图表 → experiment.tex
  method_writing:    代码 + 研究信息 → 大纲 → 递归写作 → 优化 → method.tex
  related_work:      向量DB + RAG → 逐领域逐句写作(带Judge) → related_work.tex
  introduction:      已有章节 + RAG → 大纲式写作(防虚构引用) → introduction.tex
  merge_writing:     4章合并 + 生成结论/摘要/标题 → 进化优化 → full_paper.tex

输出:
  └── result/{problem}/paper_chapter/full_paper.tex    (完整LaTeX论文)
  └── result/{problem}/paper_chapter/reference_bib.bib (参考文献)
```
