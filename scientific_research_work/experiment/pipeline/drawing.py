import json
import logging
import os
import random
from functools import partial
from multiprocessing import Pool

from langchain_community.vectorstores import Chroma

from llm.client import LLMClient
from llm.embeddings import call_Embedding_model, get_Embedding_model
from llm.rerank import reranking_intercept
from scientific_research_work.common.fs import clear_folder
from scientific_research_work.common.image import encode_image
from scientific_research_work.common.logging import setup_logging
from llm.message import image_part, system, text_part, user_content
from scientific_research_work.experiment.utils import run_tmp_script
from scientific_research_work.experiment.pipeline.analysis_perspective import text_image_separate
from scientific_research_work.experiment.prompts.drawing import (
    CHART_DRAW,
    CODE_DESCRIPTION,
    CODE_DESCRIPTION_SYSTEM,
    DRAW_CHART_SYSTEM,
    DRAW_CODE_CHECK_RESULT,
    DRAW_CODE_CHECK_SYSTEM,
    EXPERIMENTAL_INDOCATORS,
    EXPERIMENTAL_INDOCATORS_SYSTEM,
    VISUALIZATION_STRATEGY_SYSTEM,
)

random.seed(2025)


def _call_schema(model_name: str, temperature: float, system_prompt: str, parts: list[dict], response_schema):
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, response_schema=response_schema)


def _code_description(model_name: str, python_code: str) -> str:
    response = _call_schema(
        model_name,
        1,
        CODE_DESCRIPTION_SYSTEM,
        [text_part(python_code)],
        CODE_DESCRIPTION,
    )
    return response["code_description"]


def _visualization_strategy(
    model_name: str,
    python_code: str,
    text_data_list: list,
    image_data_list: list,
) -> str:
    content_parts = [text_part(f"{python_code}\n# Relevant text and image background knowledge:")]

    for data in text_data_list:
        content_parts.append(text_part(f"\n{json.dumps(data.metadata)}"))

    for data in image_data_list:
        content_parts.append(image_part(data.metadata["image_base64"], "png"))
        additional_information = {
            "save path": data.metadata.get("image_path"),
            "reference": data.metadata.get("image_references"),
        }
        content_parts.append(text_part(f"The relevant information of the reference chart or table:\n{json.dumps(additional_information)}"))

    response = _call_schema(
        model_name,
        1,
        VISUALIZATION_STRATEGY_SYSTEM,
        content_parts,
        EXPERIMENTAL_INDOCATORS,
    )
    return response["suitable_drawing_chart"]


def _draw_code(
    model_name: str,
    temperature: float,
    draw_prompt: str,
    plot_chart_template: list,
    save_path: str,
    draw_code_path: str,
) -> str:
    messages = [
        system(DRAW_CHART_SYSTEM.format(SAVE_PATH="./charts", DRAW_DATA_PATH="")),
        user_content([text_part(draw_prompt)]),
    ]

    for chart in plot_chart_template:
        messages[1]["content"].append(image_part(chart, "png"))

    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        client = LLMClient()
        response = client.call(
            messages,
            model=model_name,
            temperature=temperature,
            response_schema=CHART_DRAW,
        )
        draw_code = response["PaintCode"]
        draw_code = f"import os\nos.chdir(os.path.dirname(os.path.abspath(__file__)))\n{draw_code}"

        result = run_tmp_script(draw_code, draw_code_path)
        if result.returncode != 0:
            messages.append(
                user_content(
                    [
                        text_part(
                            f"Previous code attempt failed. Error: {result.stderr}\nPlease generate corrected code to fulfill the demand"
                        )
                    ]
                )
            )
            clear_folder(save_path)
            retry_count += 1
            continue

        logging.info("draw code success")
        return draw_code

    raise RuntimeError("draw code failed")


def _draw_code_check(model_name: str, python_code: str, chart_save_path: str) -> str:
    content_parts = [text_part(python_code)]

    if os.path.exists(chart_save_path):
        for chart_path in os.listdir(chart_save_path):
            full_path = os.path.join(chart_save_path, chart_path)
            try:
                content_parts.append(image_part(encode_image(full_path), "png"))
            except Exception:
                continue

    response = _call_schema(
        model_name,
        1,
        DRAW_CODE_CHECK_SYSTEM,
        content_parts,
        DRAW_CODE_CHECK_RESULT,
    )
    return response["draw_code_check_result"]


def draw_action(
    draw_data_path: str,
    chart_save_path: str,
    draw_code_path: str,
    experimental_code_path: str,
    analysis_explain: dict | None,
    config: dict,
) -> None:
    if config["embedding_model_source"] == "local":
        embedding_model = get_Embedding_model(config["embedding_model"], config["database_device"])
    else:
        embedding_model = call_Embedding_model(config["embedding_model"])

    vectordb_plot = Chroma(
        persist_directory=config["PlotCode_vb_path"],
        embedding_function=embedding_model,
    )

    with open(experimental_code_path, "r", encoding="utf-8") as f:
        experimental_code = f.read()

    if not analysis_explain:
        vectordb_paper = Chroma(
            persist_directory=config["experiment_persist_directory_path"],
            embedding_function=embedding_model,
        )

        code_description = _code_description(config["text_model_name"], experimental_code)

        refer_data = vectordb_paper.similarity_search(code_description, config["retrieval_template_num"])
        refer_data = reranking_intercept(
            code_description,
            refer_data,
            1,
            config["reranker_path"],
            config["database_device"],
            config["reranker_model_source"],
        )
        text_refer_data, image_refer_data = text_image_separate(refer_data)

        analysis_explain = {}
        analysis_explain["suitable_drawing_chart"] = _visualization_strategy(
            config["multimodel_model_name"],
            experimental_code,
            text_refer_data,
            image_refer_data,
        )

    refer_data = vectordb_plot.similarity_search(
        analysis_explain["suitable_drawing_chart"], config["retrieval_template_num"]
    )
    refer_data = reranking_intercept(
        analysis_explain["suitable_drawing_chart"],
        refer_data,
        1,
        config["reranker_path"],
        config["database_device"],
        config["reranker_model_source"],
    )

    plot_chart_template = []
    plot_code_template = ""
    for data_template in refer_data:
        plot_code_template = f'```\n{data_template.metadata["plot_code"]}\n```\n'
        for chart in json.loads(data_template.metadata["charts"]):
            plot_chart_template.append(encode_image(chart))

    all_draw_data = []
    for data_name in os.listdir(draw_data_path):
        all_draw_data.append(os.path.join("./experimental_result_data", data_name))

    draw_prompt = f"""
# The path for saving all experimental result data used for drawing is as follows(**The following experimental results must be used as plotting data, and random generation of plotting data is strictly prohibited.**):
{json.dumps(all_draw_data)}
# Experimental code:
```
{experimental_code}
```
# Expected visualization strategy:
{analysis_explain['suitable_drawing_chart']}
# Drawing code template:
{plot_code_template}
The result of the drawing code template is shown below:
"""

    max_try_num = config.get("max_try", 5)
    try_count = 0
    while try_count < max_try_num:
        draw_code = _draw_code(
            config["multimodel_model_name"],
            config["temperature"],
            draw_prompt,
            plot_chart_template,
            chart_save_path,
            draw_code_path,
        )

        check_result = _draw_code_check(config["multimodel_model_name"], f"{draw_prompt}\n```\n{draw_code}\n```", chart_save_path)
        if check_result == "incorrect":
            try_count += 1
            clear_folder(os.path.join(chart_save_path))
            continue
        break

    if try_count >= max_try_num:
        raise RuntimeError(f"draw_action failed for {draw_data_path}")


def drawing(config: dict, perspectives: list, log_file_name: str | None = None) -> None:
    analysis_perspective_path = config.get("analysis_perspective_path") or os.path.join(
        os.path.dirname(config["need_draw_perspectives"]), "analysis_perspective_explain.json"
    )

    with open(analysis_perspective_path, "r", encoding="utf-8") as f:
        analysis_perspective_explain = json.load(f)

    if log_file_name:
        pool_kwargs = {"initializer": setup_logging, "initargs": (log_file_name,)}
    else:
        pool_kwargs = {}

    with Pool(processes=5, **pool_kwargs) as pool:
        task_list = []
        for result_perspective in perspectives:
            if result_perspective in analysis_perspective_explain:
                analysis_explain = analysis_perspective_explain[result_perspective]
            else:
                analysis_explain = None

            result_perspective_path = os.path.join(config["need_draw_perspectives"], result_perspective)
            task_list.append(
                pool.apply_async(
                    draw_action,
                    args=(
                        os.path.join(result_perspective_path, "experimental_result_data"),
                        os.path.join(result_perspective_path, "charts"),
                        os.path.join(result_perspective_path, "draw_code.py"),
                        os.path.join(result_perspective_path, "experimental_code.py"),
                        analysis_explain,
                        config,
                    ),
                )
            )

        for task in task_list:
            task.get()

    logging.info("drawing finished")


def doing_makedirs(config: dict, perspectives: list) -> None:
    if not os.path.exists(config["log_path"]):
        os.makedirs(config["log_path"])

    for perspective in perspectives:
        perspective_dir_path = os.path.join(config["need_draw_perspectives"], perspective)
        if not os.path.exists(perspective_dir_path):
            os.makedirs(perspective_dir_path, exist_ok=True)
