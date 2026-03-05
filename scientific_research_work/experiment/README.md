# 实验模块（scientific_research_work/experiment）

本模块负责“实验流程”的完整链路：
- 生成分析视角（analysis）
- 生成实验代码与结果数据（doing）
- 生成绘图代码与图表（draw）
- 生成实验写作与 LaTeX（write）

所有入口均从 `config/config.yaml` 与 `config/problems/*.yaml` 读取参数，不依赖 `agent`。

## 目录结构
```
scientific_research_work/experiment/
  __init__.py                 # 对外入口：run_analysis/run_doing/run_draw/run_write/run_all
  utils.py                    # 临时脚本执行工具
  pipeline/
    analysis_perspective.py   # 分析视角生成
    experiment_generation.py  # 实验生成与执行
    drawing.py                # 绘图生成
    writing.py                # 实验写作与 LaTeX 集成
    orchestrator.py           # 统一调度入口
```

## 入口命令（推荐）
> 建议直接使用环境里的 python，可避免 `conda run` 在 Windows 上的 GBK 编码问题。

```powershell
F:\anaconda3\envs\auto_writing\python.exe -c "from scientific_research_work.experiment import run_analysis; run_analysis()"
F:\anaconda3\envs\auto_writing\python.exe -c "from scientific_research_work.experiment import run_doing; run_doing()"
F:\anaconda3\envs\auto_writing\python.exe -c "from scientific_research_work.experiment import run_draw; run_draw()"
F:\anaconda3\envs\auto_writing\python.exe -c "from scientific_research_work.experiment import run_write; run_write()"
```

如需指定 problem/config：
```powershell
F:\anaconda3\envs\auto_writing\python.exe -c "from scientific_research_work.common.settings import load_experiment_config; from scientific_research_work.experiment import run_analysis; run_analysis(load_experiment_config(problem_id='demo'))"
```

## 配置来源
- `config/config.yaml`
- `config/problems/<problem_id>.yaml`

关键字段示例：
- `experiment.text_model_name / multimodal_model_name`
- `experiment.embedding_model / reranker_path`
- `experiment.analysis_perspective_num`
- `drawing.max_try / retrieval_template_num`
- `experiment_writing.integration_model_name`

## 主要输出
以 `problem_id = harmony` 为例：
```
result/harmony/
  log/experimenting/*.log
  log/drawing/*.log
  log/experiment_writing/*.log
  pipeline_process_data/experimenting/get/analysis_perspective_explain.json
  pipeline_process_data/experimenting/doing/perspective/<analysis_perspective_xxx>/
    experimental_code.py
    experimental_result_data/
    charts/
```

## 关于 analysis_perspective_explain.json 的扩展字段
生成的分析视角会在原字段基础上附加以下辅助字段（用于绘图/后续流程）：
- `other_is_refer`：是否使用检索材料生成视角
- `suitable_drawing_chart`：推荐的图表类型与绘图方案
- `visualization_is_refer`：可视化建议是否参考检索材料
- `no_line_tendency`：是否倾向避免折线图

如需保持原始 5 字段，可在 `analysis_perspective.py` 中关闭附加写入逻辑。

## 常见问题
1) **run_doing 失败：找不到数据文件**
- 需要确保 `problems/<problem_id>/experimental_environment/dataset/` 下存在实验数据（例如 tma_both_cleaned_*.csv）。

2) **run_doing 失败：harm_Ours 未定义**
- 说明 LLM 生成的实验脚本没有包含方法定义，需在提示中强调必须包含/复用方法实现。

3) **run_write 失败：divide chart failed**
- 说明生成的 LaTeX 与实际图表列表不一致（缺图/路径不一致/图表过多被省略）。
- 可改为只校验交集或自动补图。

## 备注
- 绘图模块会读取 `analysis_perspective_explain.json` 的 `suitable_drawing_chart`。
- 所有 LLM 调用均通过 `llm` 模块封装接口完成。
