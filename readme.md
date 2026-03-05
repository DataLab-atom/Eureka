
# Eureka

**Eureka** is an end-to-end automated scientific paper generation system. Given a research problem, reference code, and a seed bibliography, Eureka autonomously crawls literature, designs and runs experiments, and writes a complete, citation-rich scientific paper in LaTeX.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Adding a New Problem](#adding-a-new-problem)
- [Output](#output)
- [FAQ](#faq)

---

## Overview

Eureka automates the full research paper lifecycle:

| Step | What it does |
|------|-------------|
| **Literature Database** | Crawls and indexes papers from Semantic Scholar, arXiv, and Unpaywall into a local vector database |
| **Experiment Design** | Generates multiple analytical perspectives and writes Python experiment scripts for each |
| **Experiment Execution** | Runs generated scripts (CPU/GPU parallel), produces result charts |
| **Paper Writing** | Writes methodology, related work, introduction, and experiment sections using retrieval-augmented LLM generation |
| **Paper Assembly** | Merges all sections, generates title/abstract/conclusion, applies evolutionary optimization for quality |

---

## Pipeline

```
database_building
      │
      ▼
  experiment
      │
      ▼
method_writing
      │
      ▼
related_work_writing
      │
      ▼
introduction_writing
      │
      ▼
 merge_writing  ──► result/{problem}/paper_chapter/full_paper.tex
```

Each stage saves its outputs to `result/{problem_id}/` and can be re-run independently.

---

## Requirements

- Python 3.10+
- API keys for:
  - OpenAI (or compatible LLM API)
  - Semantic Scholar
  - SiliconFlow (for BAAI/bge-m3 embeddings and reranking)

---

## Installation

```bash
git clone https://github.com/your-org/eureka.git
cd eureka
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
RETRIEVE_URL=https://api.siliconflow.cn/v1
RERANK_RETRIEVE_KEY=your-siliconflow-key
S2_API_KEY=your-semantic-scholar-key
```

---

## Configuration

Primary configuration lives in `config/config.yaml`. Key settings:

```yaml
active_problem: demo          # Problem ID to run

llm:
  temperature: 1.0
  text_model_name: gpt-4.1-mini
  multimodal_model_name: gpt-4.1-mini

experiment:
  perspectives: 5             # Number of analysis angles
  devices: [cpu]              # ["cpu"] or ["cuda:0", "cuda:1", ...]

merge_writing:
  evol_iter: 3                # Evolutionary optimization iterations
  population: 5               # Candidate paper variants per iteration
```

Each problem has its own override file at `config/problems/{problem_id}.yaml`:

```yaml
problem_id: harmony
bib_path: problems/harmony/reference.bib
```

---

## Quick Start

### Run the full pipeline

```bash
python main.py
```

### Run specific stages only

```bash
python main.py --steps database_building,experiment
python main.py --steps method_writing,related_work_writing,introduction_writing,merge_writing
```

### Switch problems without editing config

```bash
python main.py --problem my_problem
```

### List available pipeline stages

```bash
python main.py --list-steps
```

---

## Project Structure

```
Eureka/
├── main.py                            # CLI entry point
├── config/
│   ├── config.yaml                    # Global configuration
│   └── problems/
│       └── demo.yaml                  # Per-problem overrides
├── llm/                               # LLM client, embeddings, reranking
│   ├── client.py
│   ├── embeddings.py
│   └── rerank.py
├── plot_code_template/                # 12 chart style templates
├── problems/
│   └── {problem_id}/
│       ├── experiment_code.py         # Reference implementation
│       ├── experimental_data_explain.md
│       └── reference.bib             # Seed bibliography
├── result/                            # All outputs (auto-generated)
│   └── {problem_id}/
│       ├── log/                       # Per-stage logs
│       ├── paper_chapter/             # Generated .tex files
│       └── pipeline_process_data/    # Intermediate data
├── vector_database/                   # Chroma vector DBs (auto-generated)
└── scientific_research_work/
    ├── pipeline.py                    # Stage orchestrator
    ├── common/                        # Shared utilities, paper crawler
    ├── database_building/             # Literature crawl & vector DB
    ├── experiment/                    # Perspective generation & execution
    ├── method_writing/                # Methodology section
    ├── related_work_writing/          # Related work section
    ├── introduction_writing/          # Introduction section
    └── merge_writing/                 # Final assembly & optimization
```

---

## Adding a New Problem

1. **Create the problem directory:**
   ```
   problems/{your_problem_id}/
   ├── experiment_code.py          # Your reference method implementation
   ├── experimental_data_explain.md # Describe your datasets
   └── reference.bib               # Seed references (BibTeX)
   ```

2. **Create a problem config:**
   ```yaml
   # config/problems/{your_problem_id}.yaml
   problem_id: your_problem_id
   bib_path: problems/{your_problem_id}/reference.bib
   ```

3. **Run:**
   ```bash
   python main.py --problem your_problem_id
   ```

---

## Output

After a full run, the final paper is located at:

```
result/{problem_id}/paper_chapter/full_paper.tex
result/{problem_id}/paper_chapter/full_paper.bib
```

Compile to PDF:

```bash
cd result/{problem_id}/paper_chapter/
pdflatex full_paper.tex
bibtex full_paper
pdflatex full_paper.tex
pdflatex full_paper.tex
```

Intermediate outputs:

| Path | Content |
|------|---------|
| `result/{id}/paper_chapter/method.tex` | Methodology section |
| `result/{id}/paper_chapter/related_work.tex` | Related work section |
| `result/{id}/paper_chapter/introduction.tex` | Introduction section |
| `result/{id}/pipeline_process_data/perspectives.json` | Generated analysis perspectives |
| `result/{id}/pipeline_process_data/exp_N/` | Code and charts for each experiment |
| `result/{id}/log/` | Timestamped logs per stage |

---

## FAQ

**Q: Can I use a model other than GPT-4.1-mini?**
Set `text_model_name` and `multimodal_model_name` in `config.yaml`. Any OpenAI-compatible model endpoint is supported via `OPENAI_BASE_URL`.

**Q: Can I resume from a failed stage?**
Yes. Use `--steps stage_name` to re-run only the failed stage. Prior stages' outputs are preserved on disk.

**Q: How much does a full run cost?**
Costs vary by problem complexity. A typical run with 5 experiment perspectives and 3 evolutionary iterations consumes several million tokens.

**Q: Does it produce compilable LaTeX?**
The evolutionary optimization loop specifically improves LaTeX validity. However, manual review before compilation is recommended.

**Q: Can I run experiments on GPU?**
Set `experiment.devices` to a list of CUDA device strings, e.g., `["cuda:0", "cuda:1"]`.
