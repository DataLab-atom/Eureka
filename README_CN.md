# Eureka

**Eureka** 是一个端到端的科学论文自动生成系统。给定一个研究问题、参考代码和种子参考文献，Eureka 能够自主爬取文献、设计并执行实验，最终生成一篇完整的、带有引用的 LaTeX 格式科学论文。

---

## 目录

- [项目简介](#项目简介)
- [流水线流程](#流水线流程)
- [环境要求](#环境要求)
- [安装](#安装)
- [配置说明](#配置说明)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [添加新研究问题](#添加新研究问题)
- [输出结果](#输出结果)
- [常见问题](#常见问题)

---

## 项目简介

Eureka 将科学论文的完整生命周期自动化：

| 阶段 | 功能描述 |
|------|---------|
| **文献数据库构建** | 从 Semantic Scholar、arXiv、Unpaywall 爬取并索引论文，构建本地向量数据库 |
| **实验设计** | 生成多角度分析视角，为每个视角自动编写 Python 实验脚本 |
| **实验执行** | 运行生成的实验脚本（支持 CPU/GPU 并行），产出结果图表 |
| **论文写作** | 使用检索增强的 LLM 生成方法、相关工作、引言和实验章节 |
| **论文汇总** | 合并所有章节，生成标题/摘要/结论，并通过进化优化提升质量 |

---

## 流水线流程

```
database_building（数据库构建）
          │
          ▼
    experiment（实验）
          │
          ▼
  method_writing（方法写作）
          │
          ▼
related_work_writing（相关工作写作）
          │
          ▼
introduction_writing（引言写作）
          │
          ▼
  merge_writing（论文汇总）──► result/{problem}/paper_chapter/full_paper.tex
```

每个阶段将输出保存到 `result/{problem_id}/`，支持单独重跑某一阶段。

---

## 环境要求

- Python 3.10+
- 以下服务的 API 密钥：
  - OpenAI（或兼容的 LLM API）
  - Semantic Scholar
  - SiliconFlow（用于 BAAI/bge-m3 嵌入模型和重排序）

---

## 安装

```bash
git clone https://github.com/your-org/eureka.git
cd eureka
pip install -r requirements.txt
```

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
RETRIEVE_URL=https://api.siliconflow.cn/v1
RERANK_RETRIEVE_KEY=你的-siliconflow-密钥
S2_API_KEY=你的-semantic-scholar-密钥
```

---

## 配置说明

主配置文件位于 `config/config.yaml`，核心配置项如下：

```yaml
active_problem: demo          # 当前运行的问题 ID

llm:
  temperature: 1.0
  text_model_name: gpt-4.1-mini
  multimodal_model_name: gpt-4.1-mini

experiment:
  perspectives: 5             # 分析视角数量
  devices: [cpu]              # ["cpu"] 或 ["cuda:0", "cuda:1", ...]

merge_writing:
  evol_iter: 3                # 进化优化迭代轮数
  population: 5               # 每轮候选论文变体数量
```

每个问题可在 `config/problems/{problem_id}.yaml` 中覆盖配置：

```yaml
problem_id: harmony
bib_path: problems/harmony/reference.bib
```

---

## 快速开始

### 运行完整流水线

```bash
python main.py
```

### 只运行指定阶段

```bash
python main.py --steps database_building,experiment
python main.py --steps method_writing,related_work_writing,introduction_writing,merge_writing
```

### 不修改配置文件，直接切换问题

```bash
python main.py --problem my_problem
```

### 列出所有可用流水线阶段

```bash
python main.py --list-steps
```

---

## 项目结构

```
Eureka/
├── main.py                            # CLI 入口
├── config/
│   ├── config.yaml                    # 全局配置
│   └── problems/
│       └── demo.yaml                  # 问题专属配置覆盖
├── llm/                               # LLM 客户端、嵌入、重排序
│   ├── client.py
│   ├── embeddings.py
│   └── rerank.py
├── plot_code_template/                # 12 种图表风格模板
├── problems/
│   └── {problem_id}/
│       ├── experiment_code.py         # 参考方法实现代码
│       ├── experimental_data_explain.md
│       └── reference.bib             # 种子参考文献
├── result/                            # 所有输出（自动生成）
│   └── {problem_id}/
│       ├── log/                       # 各阶段日志
│       ├── paper_chapter/             # 生成的 .tex 文件
│       └── pipeline_process_data/    # 中间过程数据
├── vector_database/                   # Chroma 向量数据库（自动生成）
└── scientific_research_work/
    ├── pipeline.py                    # 阶段调度器
    ├── common/                        # 公共工具、论文爬虫
    ├── database_building/             # 文献爬取与向量数据库构建
    ├── experiment/                    # 视角生成与实验执行
    ├── method_writing/                # 方法章节写作
    ├── related_work_writing/          # 相关工作章节写作
    ├── introduction_writing/          # 引言章节写作
    └── merge_writing/                 # 论文最终汇总与优化
```

---

## 添加新研究问题

1. **创建问题目录：**
   ```
   problems/{your_problem_id}/
   ├── experiment_code.py          # 你的参考方法实现
   ├── experimental_data_explain.md # 数据集描述
   └── reference.bib               # 种子参考文献（BibTeX 格式）
   ```

2. **创建问题配置文件：**
   ```yaml
   # config/problems/{your_problem_id}.yaml
   problem_id: your_problem_id
   bib_path: problems/{your_problem_id}/reference.bib
   ```

3. **运行：**
   ```bash
   python main.py --problem your_problem_id
   ```

---

## 输出结果

完整运行后，最终论文位于：

```
result/{problem_id}/paper_chapter/full_paper.tex
result/{problem_id}/paper_chapter/full_paper.bib
```

编译为 PDF：

```bash
cd result/{problem_id}/paper_chapter/
pdflatex full_paper.tex
bibtex full_paper
pdflatex full_paper.tex
pdflatex full_paper.tex
```

中间产物说明：

| 路径 | 内容 |
|------|------|
| `result/{id}/paper_chapter/method.tex` | 方法章节 |
| `result/{id}/paper_chapter/related_work.tex` | 相关工作章节 |
| `result/{id}/paper_chapter/introduction.tex` | 引言章节 |
| `result/{id}/pipeline_process_data/perspectives.json` | 生成的分析视角 |
| `result/{id}/pipeline_process_data/exp_N/` | 各实验的代码和图表 |
| `result/{id}/log/` | 各阶段带时间戳的日志 |

---

## 常见问题

**Q：可以使用 GPT-4.1-mini 以外的模型吗？**
在 `config.yaml` 中修改 `text_model_name` 和 `multimodal_model_name` 即可。通过 `OPENAI_BASE_URL` 支持任何 OpenAI 兼容的模型接口。

**Q：某阶段失败后可以从断点继续吗？**
可以。使用 `--steps 阶段名` 单独重跑失败的阶段，之前阶段的输出会保留在磁盘上。

**Q：完整运行大约花费多少 Token？**
费用取决于问题复杂度。通常情况下，5 个实验视角、3 轮进化优化的完整运行会消耗数百万 Token。

**Q：生成的 LaTeX 可以直接编译吗？**
进化优化循环会专门改善 LaTeX 的合法性。但建议在编译前进行人工审查。

**Q：如何在 GPU 上运行实验？**
将 `experiment.devices` 设置为 CUDA 设备字符串列表，例如 `["cuda:0", "cuda:1"]`。

**Q：论文爬取会不会访问付费文献？**
系统优先访问 Semantic Scholar 开放获取 → arXiv → Unpaywall 等合法开放渠道，Sci-Hub 作为最终兜底选项。请遵守当地法律法规。
