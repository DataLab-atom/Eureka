# Eureka — Technical Documentation

> Automated Scientific Paper Generation System

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Module Reference](#2-module-reference)
   - 2.1 [Entry Point — `main.py`](#21-entry-point--mainpy)
   - 2.2 [Pipeline Orchestrator](#22-pipeline-orchestrator)
   - 2.3 [LLM Client](#23-llm-client)
   - 2.4 [Database Building](#24-database-building)
   - 2.5 [Experiment](#25-experiment)
   - 2.6 [Method Writing](#26-method-writing)
   - 2.7 [Related Work Writing](#27-related-work-writing)
   - 2.8 [Introduction Writing](#28-introduction-writing)
   - 2.9 [Merge Writing](#29-merge-writing)
   - 2.10 [Common Utilities](#210-common-utilities)
3. [Data Flow](#3-data-flow)
4. [Configuration Reference](#4-configuration-reference)
5. [Environment Variables](#5-environment-variables)
6. [Directory Layout (Runtime)](#6-directory-layout-runtime)
7. [External Services & APIs](#7-external-services--apis)
8. [Key Algorithms & Design Decisions](#8-key-algorithms--design-decisions)
9. [Extending the System](#9-extending-the-system)
10. [Known Limitations](#10-known-limitations)

---

## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           main.py (CLI)                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│               scientific_research_work/pipeline.py               │
│          Sequential execution of 6 pipeline stages               │
└──┬──────────┬───────────┬──────────┬─────────────┬──────────────┘
   │          │           │          │             │
   ▼          ▼           ▼          ▼             ▼
[DB Build] [Experiment] [Method] [Related Work] [Intro] → [Merge]
   │          │           │          │             │         │
   └──────────┴───────────┴──────────┴─────────────┴────▶ result/
                                                         {problem}/
```

### Design Principles

- **Sequential pipeline**: Each stage depends on outputs of prior stages. Stages are individually re-runnable by passing `--steps`.
- **LLM-centric generation**: Every writing and analysis step is powered by an `LLMClient` wrapper around OpenAI-compatible APIs.
- **Retrieval-augmented generation (RAG)**: A Chroma vector database, built from crawled papers, supplies relevant context to all writing stages.
- **Evolutionary optimization**: The final merge stage runs an LLM-based judge-and-evolve loop to iteratively improve paper quality.

---

## 2. Module Reference

### 2.1 Entry Point — `main.py`

**Location:** `main.py`

**CLI Interface:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--steps` | `str` | all steps | Comma-separated subset of pipeline steps to run |
| `--config` | `str` | `config/config.yaml` | Path to config YAML |
| `--problem` | `str` | value in config | Override the active problem ID |
| `--list-steps` | flag | — | Print available steps and exit |

**Behavior:**
1. Loads config YAML and (optionally) problem-specific YAML.
2. Validates requested steps against `PIPELINE_STEPS`.
3. Calls `run_pipeline(steps, config)`.

---

### 2.2 Pipeline Orchestrator

**Location:** `scientific_research_work/pipeline.py`

**Constants:**
```python
PIPELINE_STEPS = (
    "database_building",
    "experiment",
    "method_writing",
    "related_work_writing",
    "introduction_writing",
    "merge_writing",
)
```

**Public API:**

```python
def run_pipeline(steps: list[str], config: dict) -> None
```

Iterates over the requested steps in order, dynamically imports the matching module's `run()` function, and calls it with the config dict. Logs start/end time and any exceptions.

---

### 2.3 LLM Client

**Location:** `llm/client.py`

**Classes and Functions:**

```python
def get_client(api_key: str, base_url: str | None) -> openai.OpenAI
```

```python
class LLMClient:
    def __init__(self, model: str, temperature: float, log_dir: str)
    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        schema: type[BaseModel] | None = None,
        retries: int = 3,
    ) -> str | BaseModel
```

**Token Logging:**
Each `LLMClient` instance writes cumulative token counts to:
- `{log_dir}/input_token.log`
- `{log_dir}/output_token.log`

**Retry Logic:**
On API errors, retries up to `retries` times with a 2-second fixed delay.

**JSON Mode:**
- If `schema` is provided, uses OpenAI structured output (Pydantic model).
- If `json_mode=True`, enables `response_format={"type": "json_object"}` and falls back to parsing JSON from markdown fences on failure.

---

### 2.4 Database Building

**Location:** `scientific_research_work/database_building/`

**Entry:** `core.py :: run(config)`

#### Sub-pipeline

```
1. resolve_paths()          — Set up result directory structure
2. crawl_papers()           — Query Semantic Scholar for related papers
3. download_pdfs()          — Fetch PDFs via arXiv / Unpaywall / Sci-Hub
4. pdf_to_markdown()        — Convert PDFs → Markdown (local or API mode)
5. md_postprocess()         — Clean and chunk Markdown text
6. build_paper_vector_db()  — Embed and store in Chroma (paper texts + images)
7. build_plot_code_db()     — Embed plot code templates for drawing retrieval
```

#### Vector Database Schema

**Paper Vector DB** (`vector_database/{problem_id}/paper/`):
| Field | Type | Description |
|-------|------|-------------|
| `page_content` | `str` | Text chunk or image description |
| `metadata.source` | `str` | Paper title |
| `metadata.type` | `str` | `"text"` or `"image"` |
| `metadata.bib_key` | `str` | BibTeX citation key |

**Plot Code Vector DB** (`vector_database/{problem_id}/plot_code/`):
| Field | Type | Description |
|-------|------|-------------|
| `page_content` | `str` | Chart description summary |
| `metadata.template_id` | `str` | Template folder name |
| `metadata.chart_file` | `str` | Source Python file |

#### PDF Extraction Modes

| Mode | Description |
|------|-------------|
| `local` | Uses local PDF parsing library |
| `api` | Calls a remote PDF-to-Markdown API endpoint |

Configured via `database_building.pdf_transform_mode` in `config.yaml`.

---

### 2.5 Experiment

**Location:** `scientific_research_work/experiment/`

**Entry:** `pipeline/orchestrator.py :: experiment(config, mode)`

#### Modes

| Mode | Function called | Description |
|------|-----------------|-------------|
| `get` | `run_analysis()` | Generate analysis perspectives only |
| `doing` | `run_doing()` | Execute experiments for each perspective |
| `draw` | `run_draw()` | Generate charts for completed experiments |
| `write` | `run_write()` | Write experiment descriptions |
| `all` | `run_all()` | Full sub-pipeline (default) |

#### Analysis Perspective Generation (`analysis_perspective.py`)

1. Reads the problem's `experiment_code.py` and bibliography.
2. Queries the paper vector DB for background context.
3. Uses structured LLM output to produce:
   - **Method name & statement** (what the algorithm does)
   - **N analysis perspectives** (each is an experimental angle, e.g., ablation study, comparison with baseline, hyperparameter sensitivity)
   - **Visualization strategy** per perspective (chart type recommendation)

**Output:** `result/{problem}/pipeline_process_data/perspectives.json`

#### Experiment Code Generation (`experiment_generation.py`)

For each perspective:
1. LLM generates a complete Python experiment script.
2. Script is validated by a **judge agent** checking:
   - Adherence to the analysis perspective
   - Correct reproduction of the core method
   - Proper loading/use of experimental data
3. On failure, LLM performs error-guided correction (up to 3 retries).
4. Approved scripts are executed via `multiprocessing` with device assignment (CPU/GPU).

**Output per perspective:** `result/{problem}/pipeline_process_data/exp_{i}/`
```
exp_{i}/
├── code.py          # Generated experiment script
├── output/          # stdout / stderr
└── charts/          # Saved figure files
```

#### Drawing (`drawing.py`)

- Retrieves the top-K most visually similar plot code templates from the plot code vector DB.
- LLM rewrites templates to match the current experiment's data and perspective.
- Outputs polished chart files alongside experiment results.

---

### 2.6 Method Writing

**Location:** `scientific_research_work/method_writing/`

**Entry:** `core.py :: write_method(config)`

#### Algorithm

```
1. Generate hierarchical outline (section → subsection → paragraph tasks)
2. For each task (BFS order):
   a. "think" step  — LLM reasons about what to write
   b. "write" step  — LLM produces LaTeX paragraph
3. Integration step — Merge all task outputs into coherent section
4. Improvement rounds (configurable, default=2):
   a. LLM identifies weak points
   b. LLM rewrites those passages
5. Typesetting — Apply LaTeX formatting and embed experiment figures
```

#### `TaskMemory` Class

Tracks the writing state as a tree:
```python
@dataclass
class TaskMemory:
    task_id: str
    task_type: Literal["think", "write"]
    context: str          # retrieved related content
    result: str           # LLM output
    children: list[TaskMemory]
```

**Output:** `result/{problem}/paper_chapter/method.tex`

---

### 2.7 Related Work Writing

**Location:** `scientific_research_work/related_work_writing/`

**Entry:** `core.py :: write_related_work(config)`

#### Algorithm

```
1. Identify research fields from method code (LLM)
2. For each field:
   a. Query paper vector DB (top-K retrieval)
   b. Rerank results with FlagEmbedding cross-encoder
   c. LLM writes a field-specific paragraph with inline citations
3. Integration — Merge field paragraphs, remove duplicate citations
4. Citation validation — Verify every \cite{} key exists in .bib
5. Bibliography merge — Combine problem .bib with crawled paper entries
```

**Output:**
- `result/{problem}/paper_chapter/related_work.tex`
- `result/{problem}/paper_chapter/related_work.bib`

---

### 2.8 Introduction Writing

**Location:** `scientific_research_work/introduction_writing/`

**Entry:** `core.py :: write_introduction(config)`

#### Algorithm

```
1. Read existing chapters: related_work.tex, method.tex, experiment summaries
2. Generate hierarchical outline for the introduction
3. Execute tasks (same think → write pattern as method_writing)
4. Validate and format citations (both \cite{} and figure references)
5. Integration and improvement rounds
```

Notably, introduction writing reads all other completed sections to ensure the introduction accurately summarizes the full paper.

**Output:** `result/{problem}/paper_chapter/introduction.tex`

---

### 2.9 Merge Writing

**Location:** `scientific_research_work/merge_writing/`

**Entry:** `core.py :: merge_writing(config)`

#### Algorithm

```
1. Collect all chapters:
   introduction.tex / related_work.tex / method.tex / experiment/*.tex
2. Generate title (LLM)
3. Generate abstract (LLM, conditioned on all chapters)
4. Generate conclusion (LLM)
5. Merge into full paper LaTeX template
6. Improvement rounds (global):
   a. LLM identifies structural/logical issues
   b. LLM patches specific passages
7. LaTeX evolution loop (configurable iterations × population size):
   a. Population of paper variants is maintained
   b. Judge LLM scores each variant on multiple criteria
   c. Top variants produce the next generation via LLM mutation
8. Final submission check — Ensure all \cite{} keys resolve, figures exist
```

#### Evolution Criteria (Judge LLM)

- Logical coherence and narrative flow
- Citation density and relevance
- Contribution clarity
- Experiment–claim alignment
- LaTeX compilability

**Output:**
- `result/{problem}/paper_chapter/full_paper.tex`
- `result/{problem}/paper_chapter/full_paper.bib`

---

### 2.10 Common Utilities

#### `common/settings.py`

Central configuration loader. Merges:
1. `config/config.yaml` (base)
2. `config/problems/{problem_id}.yaml` (problem override)
3. Environment variables

Key functions:
```python
load_base_config(config_path, problem_id) -> dict
load_database_config(base_config) -> DatabaseConfig
load_experiment_config(base_config) -> ExperimentConfig
load_writing_config(base_config) -> WritingConfig
```

#### `common/paper_crawler/`

Multi-source paper download pipeline:

| Source | Class | Notes |
|--------|-------|-------|
| Semantic Scholar | `SemanticScholarAgent` | Metadata + abstract |
| arXiv | `ArXivMethod` | Open access PDFs |
| Unpaywall | `UnpaywallMethod` | Open access PDFs |
| Sci-Hub | `SciHubModule` | Fallback download |

Includes a CAPTCHA-handling agent for Sci-Hub access.

#### `common/judge_evol/`

Provides the judge-and-evolve framework used in `merge_writing`:
- `agent_as_a_judge/` — LLM-based scoring agents
- Evolution loop utilities

#### `llm/embeddings.py`

Abstracts embedding model selection:
- **BAAI/bge-m3** via SiliconFlow API (default)
- **OpenAI embeddings** (optional)

#### `llm/rerank.py`

Wraps FlagEmbedding cross-encoder for document reranking in retrieval steps.

---

## 3. Data Flow

```
problems/{id}/
├── experiment_code.py          → [Experiment] → exp results
├── experimental_data_explain.md→ [Experiment] → analysis context
├── reference.bib               → [DB Build]  → crawl seeds
│                               → [All Writing] → citation pool
└── template/                   → [Method Writing] → LaTeX template

                ↓ Database Building
vector_database/{id}/
├── paper/                      → RAG retrieval for all writing stages
└── plot_code/                  → Chart template retrieval

                ↓ Experiment
result/{id}/pipeline_process_data/
├── perspectives.json
└── exp_{i}/code.py, charts/

                ↓ Writing stages
result/{id}/paper_chapter/
├── method.tex
├── related_work.tex + .bib
├── introduction.tex
├── experiment_{i}.tex
└── full_paper.tex + .bib      ← Final output
```

---

## 4. Configuration Reference

### `config/config.yaml`

```yaml
active_problem: demo          # Which problem config to load

paths:
  result_dir: result
  problems_dir: problems
  vector_db_dir: vector_database
  plot_code_template_dir: plot_code_template

llm:
  temperature: 1.0
  text_model_name: gpt-4.1-mini
  multimodal_model_name: gpt-4.1-mini

database_building:
  embedding_model: BAAI/bge-m3
  pdf_transform_mode: api       # "local" | "api"
  text_data_processing: true

experiment:
  mode: all                     # "get"|"doing"|"draw"|"write"|"all"
  devices: [cpu]                # list of "cpu" or "cuda:N"
  perspectives: 5               # number of analysis angles
  reranker_path: ""             # local path to reranker model (empty = API)
  retrieval_data_num: 20        # top-K from vector DB
  rerank_data_num: 8            # top-K after reranking

drawing:
  retrieval_template_num: 3     # chart templates to retrieve
  max_try: 3                    # retry attempts for drawing

method_writing:
  write_model: gpt-4.1-mini
  integration_model: gpt-4.1-mini
  temperature: 1.0
  improve_round: 2

related_work_writing:
  text_model: gpt-4.1-mini
  retrieval_data_num: 5
  used_cite_num: 3
  field_num: 3

introduction_writing:
  text_model: gpt-4.1-mini
  retrieval_data_num: 8
  used_data_num: 5

merge_writing:
  write_model: gpt-4.1-mini
  judge_model: gpt-4.1-mini
  improve_round: 2
  evol_iter: 3
  population: 5
```

### `config/problems/{id}.yaml`

```yaml
problem_id: harmony
bib_path: problems/harmony/reference.bib
```

---

## 5. Environment Variables

Loaded from `.env` file in the project root via `llm/env.py`.

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI / compatible API key |
| `OPENAI_BASE_URL` | No | Custom API base URL (e.g., Azure, local proxy) |
| `RETRIEVE_URL` | Yes | Embedding API endpoint (SiliconFlow or compatible) |
| `RERANK_RETRIEVE_KEY` | Yes | API key for reranking service |
| `S2_API_KEY` | Yes | Semantic Scholar API key |

**`.env` file format:**
```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
RETRIEVE_URL=https://api.siliconflow.cn/v1
RERANK_RETRIEVE_KEY=...
S2_API_KEY=...
```

---

## 6. Directory Layout (Runtime)

After a full pipeline run for problem `demo`:

```
result/
└── demo/
    ├── log/
    │   ├── database_building/YYYYMMDD_HHMMSS.log
    │   ├── experiment/YYYYMMDD_HHMMSS.log
    │   ├── method_writing/YYYYMMDD_HHMMSS.log
    │   ├── related_work_writing/YYYYMMDD_HHMMSS.log
    │   ├── introduction_writing/YYYYMMDD_HHMMSS.log
    │   └── merge_writing/YYYYMMDD_HHMMSS.log
    ├── paper_chapter/
    │   ├── method.tex
    │   ├── related_work.tex
    │   ├── related_work.bib
    │   ├── introduction.tex
    │   ├── experiment_0.tex
    │   ├── experiment_1.tex
    │   ├── ...
    │   ├── full_paper.tex
    │   └── full_paper.bib
    └── pipeline_process_data/
        ├── perspectives.json
        ├── exp_0/
        │   ├── code.py
        │   ├── output/
        │   └── charts/
        └── exp_N/
            └── ...

vector_database/
└── demo/
    ├── paper/           # Chroma DB — paper text & image chunks
    └── plot_code/       # Chroma DB — plotting templates

llm/
├── input_token.log      # Cumulative input tokens
└── output_token.log     # Cumulative output tokens
```

---

## 7. External Services & APIs

| Service | Purpose | SDK / Protocol |
|---------|---------|----------------|
| OpenAI API | LLM text generation | `openai` Python SDK |
| SiliconFlow | BAAI/bge-m3 embeddings | OpenAI-compatible REST |
| SiliconFlow | FlagEmbedding reranking | OpenAI-compatible REST |
| Semantic Scholar | Paper metadata & related papers | `semanticscholar` SDK |
| arXiv | Open-access PDF download | REST API |
| Unpaywall | Open-access PDF lookup | REST API |
| Sci-Hub | PDF fallback source | HTTP scraping |
| PDF-to-Markdown API | PDF extraction (api mode) | Custom REST endpoint |

---

## 8. Key Algorithms & Design Decisions

### 8.1 Hierarchical Task Decomposition (Writing Modules)

All writing modules (method, introduction) decompose the writing problem into a tree of tasks. This approach:
- Prevents context length overflow in a single LLM call
- Enables parallel generation of independent subtasks
- Produces more focused, coherent paragraphs per task

### 8.2 Retrieval-Augmented Generation

Each writing stage retrieves relevant chunks from the Chroma vector DB using:
1. Dense retrieval (embedding similarity)
2. Cross-encoder reranking (FlagEmbedding) to improve precision

This grounds generated text in actual paper content, reducing hallucination.

### 8.3 Evolutionary Paper Optimization

The merge stage applies a genetic-algorithm-style loop:
- **Population:** multiple variants of the full paper
- **Fitness:** LLM judge scores on coherence, citation quality, contribution clarity
- **Mutation:** LLM rewrites low-scoring sections
- **Selection:** top-scoring variants survive to the next generation

This exploits LLMs' ability to evaluate and improve text iteratively.

### 8.4 Multi-Source PDF Acquisition

Papers are fetched with a priority fallback chain:
```
Semantic Scholar open access → arXiv → Unpaywall → Sci-Hub
```
This maximizes coverage while preferring legal open-access sources.

### 8.5 Device-Parallel Experiment Execution

Experiment scripts are dispatched to available devices (CPU/GPU) via Python `multiprocessing`. A simple queue assigns each perspective to the next free device, enabling parallelism proportional to available hardware.

---

## 9. Extending the System

### Adding a New Pipeline Stage

1. Create `scientific_research_work/{stage_name}/core.py` with a `run(config: dict)` function.
2. Add `"{stage_name}"` to `PIPELINE_STEPS` in `pipeline.py`.
3. Add any stage-specific config keys to `config/config.yaml` and `common/settings.py`.

### Adding a New LLM Provider

1. Edit `llm/client.py` — `get_client()` accepts `base_url`; point it at any OpenAI-compatible endpoint.
2. Update `OPENAI_BASE_URL` in `.env`.

### Adding a New Paper Source

1. Implement a method class in `common/paper_crawler/paper/method/`.
2. Register it in the crawler's source priority chain in `database_building/core.py`.

### Adding a New Problem

1. Create `problems/{id}/` with:
   - `experiment_code.py` — reference implementation
   - `experimental_data_explain.md` — dataset description
   - `reference.bib` — seed bibliography
2. Create `config/problems/{id}.yaml`:
   ```yaml
   problem_id: {id}
   bib_path: problems/{id}/reference.bib
   ```
3. Set `active_problem: {id}` in `config/config.yaml` or pass `--problem {id}`.

---

## 10. Known Limitations

| Area | Limitation |
|------|------------|
| **LaTeX compilation** | Generated `.tex` files are not automatically compiled; manual `pdflatex` run required |
| **Experiment execution** | Generated experiment code may fail if the problem environment is not properly set up |
| **Sci-Hub access** | Legal and availability varies by jurisdiction; CAPTCHA handling may fail |
| **Token cost** | A full pipeline run can consume millions of tokens across all stages |
| **Single-problem parallelism** | Stages are strictly sequential; no support for running multiple problems in parallel |
| **No formal test suite** | Correctness relies on LLM output quality and manual inspection |
| **Embedding model dependency** | BAAI/bge-m3 requires SiliconFlow API key or local deployment |
