# database_building 模块测试指南

本说明用于本地测试 `scientific_research_work/database_building` 模块。

## 1. 前置条件
- Python 环境可用（建议使用项目既有虚拟环境）
- 已安装项目依赖（包含：mineru、langchain、pydantic、semanticscholar 等）
- 可用的外部服务：
  - MinerU PDF 解析服务（需要 `PDF_MD_TOKEN`）
  - SemanticScholar API（需要 `S2_API_KEY`）
- LLM 接口（通过 `llm.LLMClient` 或 `llm.client.get_client`）

## 2. 配置检查
配置文件位置：
- `config/config.yaml`
- `config/problems/demo.yaml`

关键字段说明：
- `active_problem`: 当前问题配置（对应 `config/problems/*.yaml`）
- `paths.*`: 结果与数据路径配置
- `database_building.*`: 模块超参数（embedding、PDF 解析方式、chunk 配置等）
- `S2_API_KEY`: SemanticScholar API Key（环境变量，见 `.env`）

`config/problems/demo.yaml` 至少需要：
- `problem_id`
- `bib_path`（相对项目根目录或绝对路径）

## 3. 测试命令
在项目根目录执行（会同时构建论文向量库与绘图代码向量库）：

```powershell
python -c "from scientific_research_work.database_building import database_building; database_building()"
```

如果你只想构建绘图代码向量库（PlotCode）：

```powershell
python -c "from scientific_research_work.database_building.vector_db.build import build_plotcode_vb; from scientific_research_work.common.settings import load_database_config; build_plotcode_vb(load_database_config())"
```

如果你需要自定义配置路径：

```powershell
python -c "from scientific_research_work.common.settings import load_database_config; from scientific_research_work.database_building import database_building; database_building(load_database_config('D:/experiment_auto_write_old/config/config.yaml'))"
```

## 4. 预期输出
运行后会生成以下内容（示例路径）：
- 日志：`result/<problem_id>/log/database_building/*.log`
- PDF 与解析结果：`result/<problem_id>/pipeline_process_data/database_building/paper/...`
- 向量库：
  - `vector_database/paper/<problem_id>/experiment`
  - `vector_database/paper/<problem_id>/full_text`

## 5. 常见问题
- **PDF 解析失败**：检查 `PDF_MD_TOKEN` 是否有效；或 `database_building.pdf_transform_mode` 是否为 `api`
- **SemanticScholar 返回为空**：确认 `S2_API_KEY`；检查网络
- **LLM 调用失败**：检查 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 的环境变量配置
- **路径找不到**：确认 `config.yaml` 里的 `paths` 与 `bib_path` 是否正确

## 6. 最小验证建议
- 首次测试建议使用较小的 `all_paper_num`（例如 5）
- 确保 `config/problems/demo.yaml` 的 `bib_path` 指向一个小 bib

---
如需进一步自动化测试（例如 smoke test 脚本），可继续扩展该 README。
