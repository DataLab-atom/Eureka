import logging
import os
import json
from collections import Counter

from scientific_research_work.common.image import encode_image
from llm.client import LLMClient
from llm.message import image_part, system, text_part, user_content
from scientific_research_work.experiment.prompts.writing import (
    DIVIDE_CHART_SYSTEM,
    EXPERIMENT_GENERATE_TEX_SYSTEM,
    EXPERIMENT_INTEGRATION_WRITE_SYSTEM,
    EXPERIMENT_THINK_TASK_SYSTEM,
    EXPERIMENT_TYPESETTING_SYSTEM,
    EXPERIMENT_WRITE_OUTLINE_SYSTEM,
    EXPERIMENT_WRITE_TASK_SYSTEM,
    CHART_CAPTION_SYSTEM,
    CAPTION_CHART_REPLACE_SYSTEM,
    CAPTION,
    CAPTION_REPLACE,
    Generate_Tex,
    Integration_Write,
    Think_Task,
    Write_Task,
)


def _get_multimodal_model(config: dict) -> str:
    return (
        config.get("multimodal_model_name")
        or config.get("multimodel_model_name")
        or config.get("text_model_name")
    )


def _get_temperature(config: dict) -> float:
    return config.get("writing_temperature", config.get("temperature", 1.0))


def _call_schema(model_name: str, temperature: float, system_prompt: str, parts: list[dict], response_schema):
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, response_schema=response_schema)


def _call_json(model_name: str, temperature: float, system_prompt: str, parts: list[dict]):
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, json_mode=True)


def _build_outline_content(writing_requirement: str, experiment_add_info: list[tuple]) -> list:
    content_parts = [text_part(writing_requirement)]

    if experiment_add_info:
        content_parts.append(text_part("# The all experimental charts of results is as follows:"))

    for add_info in experiment_add_info:
        chart_path, chart_base64, experiment_name, draw_code = add_info
        if os.path.splitext(chart_path)[1].lower() == ".png":
            content_parts.append(image_part(chart_base64, "png"))
            chart_info = {
                "Draw code": draw_code,
                "Save path": chart_path,
                "From experiment": experiment_name,
            }
            content_parts.append(
                text_part(f"Information in the chart of the result of the experiment:\n{json.dumps(chart_info)}")
            )

    return content_parts


def _append_task_charts(content_parts: list, task: dict, experiment_add_info: list[tuple]) -> None:
    if not task.get("need_chart_table"):
        return

    content_parts.append(
        text_part(
            "# The experimental result chart is as follows(need to be used must be embedded in the text content according to the Markdown embedding method):"
        )
    )

    for chart_table in task["need_chart_table"]:
        for add_info in experiment_add_info:
            if add_info[0] == chart_table and os.path.splitext(add_info[0])[1].lower() == ".png":
                logging.info("写作中使用了图表 %s", add_info[0])
                content_parts.append(image_part(add_info[1], "png"))
                chart_info = {
                    "Save path": add_info[0],
                    "Draw code": add_info[3],
                    "From experiment": add_info[2],
                }
                content_parts.append(
                    text_part(f"Information in the chart of the results of experiment:{json.dumps(chart_info)}")
                )



def _append_think_charts(content_parts: list, task: dict, experiment_add_info: list[tuple]) -> None:
    if not task.get("need_chart_table"):
        return

    content_parts.append(text_part("# The experimental result chart is as follows:"))

    for chart_table in task["need_chart_table"]:
        for add_info in experiment_add_info:
            if add_info[0] == chart_table and os.path.splitext(add_info[0])[1].lower() == ".png":
                content_parts.append(image_part(add_info[1], "png"))
                chart_info = {
                    "Draw code": add_info[3],
                    "Save path": add_info[0],
                    "From experiment": add_info[2],
                }
                content_parts.append(
                    text_part(f"Information in the chart of the results of this experiment:{json.dumps(chart_info)}")
                )



def get_writing_requirement(config: dict, perspectives: list):
    with open(os.path.join(config["code_path"], "Ours.py"), "r", encoding="utf-8") as f:
        ours_code = f.read()

    with open(config["research_information"], "r", encoding="utf-8") as f:
        research_information = json.load(f)

    writing_requirement = (
        f"# {research_information['main_research']} For this,\n"
        f"## We came up with a method:**{research_information['method_name']}**."
        "The python code implementation of our method is as follows:\n"
        f"```\n{ours_code}\n```\n"
    )

    writing_requirement += "\n\n## The Python code implementation of the baseline method is as follows:\n"
    other_code_path = os.path.join(config["code_path"], "other")
    for code_idx in os.listdir(other_code_path):
        code_name = os.listdir(os.path.join(other_code_path, code_idx))[0]
        with open(os.path.join(other_code_path, code_idx, code_name), "r", encoding="utf-8") as f:
            add_content = (
                f"\n- The Python code implementation of the {code_name.split('.')[0]} baseline method is as follows:\n"
                f"```\n{f.read()}\n```\n\n"
            )
        writing_requirement += add_content

    experiment_set = {}
    for idx, perspective in enumerate(perspectives):
        with open(
            os.path.join(config["result_perspective_path"], perspective, "experimental_code.py"),
            "r",
            encoding="utf-8",
        ) as f:
            experiment_set[f"Experiment {idx+1}"] = f.read()

    with open(config["experimental_data_explain_path"], "r", encoding="utf-8") as f:
        writing_requirement += (
            "\n# At the same time, we conducted a series of experiments for this purpose.\n"
            "## The dataset used for the experiments is explained as follows:\n"
            f"{f.read()}\n"
            "## The experimental codes are as follows:\n"
            f"{json.dumps(experiment_set)}\n\n"
        )

    return writing_requirement, experiment_set


class TaskMemory:
    def __init__(self, task: dict, writing_requirement: str):
        self.task = task
        self.task_list = []
        self.task_outline = ""
        self.now_task_index = 0
        self.writing_requirement = writing_requirement
        self.done_write = ""
        self.done_write_dict = {}
        self.init_task()

    def update_task(self):
        self.now_task_index += 1

    def display_task(self) -> None:
        need_display_content = ""
        for index, task in enumerate(self.task_list):
            if index == self.now_task_index:
                need_display_content += f"*{task}\n"
            else:
                need_display_content += f"{task}\n"
        logging.info("\n" + need_display_content)

    def log_outline_result(self, config: dict):
        outline_result = (
            f"# {'-'*46} Write Outline {'-'*46}\n\n\n{self.task_outline}\n\n\n"
            f"# {'-'*47} Task Result {'-'*47}\n\n\n"
        )

        for task in self.task_list:
            task_id = task.split(":")[0].split("\t")[-1]
            space_num = len(task.split(task_id)[0])

            outline_result += (
                f"{'#'*(space_num+2)} {'&emsp;'*(space_num*2)}**{task_id}**\n"
                f"{'&emsp;'*(space_num*2+2)+'_'+task.split(f'{task_id}: ')[-1]+'_'}\n"
            )

            for key in self.done_write_dict:
                if key in task:
                    outline_result += f"```markdown\n{self.done_write_dict[key]}\n```\n"
                    break

            outline_result += "\n"

        with open(config["outline_result_path"], "w", encoding="utf-8") as f:
            f.write(outline_result)

    def init_task(self) -> None:
        def init_func(task: dict) -> None:
            space_num = ""
            if task["id"] != "root" and task["id"].count(".") == 0:
                space_num += "\t"
            elif task["id"] == "root":
                pass
            else:
                space_num += (task["id"].count(".") + 1) * "\t"

            if task.get("length"):
                if task.get("need_chart_table") and task["need_chart_table"]:
                    self.task_list.append(
                        f"{space_num}{task['id']}: {task['goal']}   (task_type: {task['task_type']}, length: {task['length']}, need_chart_table: {task['need_chart_table']})"
                    )
                    self.task_outline += (
                        f"{space_num}**{task['id']}**\n"
                        f"{space_num}: {task['goal']}   (task_type: {task['task_type']}, length: {task['length']}, need_chart_table: {task['need_chart_table']})\n\n"
                    )
                else:
                    self.task_list.append(
                        f"{space_num}{task['id']}: {task['goal']}   (task_type: {task['task_type']}, length: {task['length']})"
                    )
                    self.task_outline += (
                        f"{space_num}**{task['id']}**\n"
                        f"{space_num}: {task['goal']}   (task_type: {task['task_type']}, length: {task['length']})\n\n"
                    )
            else:
                if task.get("need_chart_table") and task["need_chart_table"]:
                    self.task_list.append(
                        f"{space_num}{task['id']}: {task['goal']}   (task_type: {task['task_type']}, need_chart_table: {task['need_chart_table']})"
                    )
                    self.task_outline += (
                        f"{space_num}**{task['id']}**\n"
                        f"{space_num}: {task['goal']}   (task_type: {task['task_type']}, need_chart_table: {task['need_chart_table']})\n\n"
                    )
                else:
                    self.task_list.append(
                        f"{space_num}{task['id']}: {task['goal']}   (task_type: {task['task_type']})"
                    )
                    self.task_outline += (
                        f"{space_num}**{task['id']}**\n"
                        f"{space_num}: {task['goal']}   (task_type: {task['task_type']})\n\n"
                    )

            if task.get("sub_tasks") and task["sub_tasks"]:
                for plan_one in task["sub_tasks"]:
                    init_func(plan_one)

        init_func(self.task)


def _build_think_prompt(taskmemory: TaskMemory, related_think: str) -> str:
    return EXPERIMENT_THINK_TASK_SYSTEM.format(
        ALL_WRITE_TASK=taskmemory.writing_requirement,
        ALL_WRITE_TASK_PLAN=json.dumps(taskmemory.task, ensure_ascii=False),
        RELATED_THINK=related_think,
        DONE_WRITE=taskmemory.done_write,
    )


def _build_write_prompt(taskmemory: TaskMemory, related_think: str) -> str:
    return EXPERIMENT_WRITE_TASK_SYSTEM.format(
        ALL_WRITE_TASK=taskmemory.writing_requirement,
        ALL_WRITE_TASK_PLAN=json.dumps(taskmemory.task, ensure_ascii=False),
        RELATED_THINK=related_think,
        DONE_WRITE=taskmemory.done_write,
    )


def write_task(taskmemory: TaskMemory, task: dict, related_think: str, experiment_add_info: list[tuple], config: dict) -> None:
    logging.info("start task: %s", task["goal"])
    taskmemory.display_task()
    taskmemory.update_task()

    temperature = _get_temperature(config)

    if task.get("need_chart_table") and task["need_chart_table"]:
        try_num = 0
        while try_num < 3:
            system_prompt = _build_write_prompt(taskmemory, related_think)
            content_parts = []
            content_parts.append(
                text_part(
                    f"# The writing subtask you need to complete is:\n {task['goal']}. Storage path of experimental result chart(need to be used must be embedded in the text content according to the Markdown embedding method):{json.dumps(task['need_chart_table'])}"
                )
            )
            _append_task_charts(content_parts, task, experiment_add_info)

            write_result = _call_schema(
                _get_multimodal_model(config),
                temperature,
                system_prompt,
                content_parts,
                Write_Task,
            )["write"]

            missing_required = False
            for chart in task["need_chart_table"]:
                if chart not in write_result and chart not in taskmemory.done_write:
                    missing_required = True
                    break

            if not missing_required:
                break

            try_num += 1
            logging.error("required chart missing, retry %s", try_num)

        if try_num >= 3:
            raise RuntimeError("required chart/table missing in write result")
    else:
        system_prompt = _build_write_prompt(taskmemory, related_think)
        content_parts = [text_part(f"# The writing subtask you need to complete is:\n {task['goal']}")]
        write_result = _call_schema(
            config["text_model_name"],
            temperature,
            system_prompt,
            content_parts,
            Write_Task,
        )["write"]

    taskmemory.done_write += write_result + "\n\n\n"
    logging.info("task done: %s", task["goal"])
    taskmemory.done_write_dict[task["goal"]] = write_result



def think_task(taskmemory: TaskMemory, task: dict, related_think: str, experiment_add_info: list[tuple], config: dict) -> str:
    logging.info("start task: %s", task["goal"])
    taskmemory.display_task()
    taskmemory.update_task()

    temperature = _get_temperature(config)

    system_prompt = _build_think_prompt(taskmemory, related_think)
    if task.get("need_chart_table") and task["need_chart_table"]:
        content_parts = [
            text_part(
                f"# The writing subtask you need to complete is:\n {task['goal']}. Storage path of experimental result chart:{json.dumps(task['need_chart_table'])}"
            )
        ]
        _append_think_charts(content_parts, task, experiment_add_info)
        model_name = _get_multimodal_model(config)
    else:
        content_parts = [text_part(f"# The writing subtask you need to complete is:\n {task['goal']}")]
        model_name = config["text_model_name"]

    think_result = _call_schema(
        model_name,
        temperature,
        system_prompt,
        content_parts,
        Think_Task,
    )["think"]

    logging.info("task done: %s", task["goal"])
    taskmemory.done_write_dict[task["goal"]] = think_result

    return think_result



def execute_task(taskmemory: TaskMemory, experiment_add_info: list[tuple], config: dict):
    def execute_step_task(task: dict, related_think: str):
        this_layer_related_think = ""

        if task.get("sub_tasks") and task["sub_tasks"]:
            logging.info("start task: %s", task["goal"])
            taskmemory.display_task()
            taskmemory.update_task()

            for task_one in task["sub_tasks"]:
                result_task = execute_step_task(task_one, related_think + this_layer_related_think)
                if result_task:
                    this_layer_related_think += result_task + "\n\n\n"

            logging.info("task done: %s", task["goal"])
        else:
            if task["task_type"] == "think":
                return think_task(taskmemory, task, related_think, experiment_add_info, config)
            if task["task_type"] == "write":
                write_task(taskmemory, task, related_think, experiment_add_info, config)

    execute_step_task(taskmemory.task, "")



def _divide_chart(model_name: str, tex_code: str) -> dict:
    response = _call_json(model_name, 1, DIVIDE_CHART_SYSTEM, [text_part(tex_code)])
    if isinstance(response, dict) and "devide_result" in response:
        return response["devide_result"]
    return response



def _generate_caption(model_name: str, chart_paths: list, experiment_add_info: list, experiment_set: dict) -> str:
    draw_code = ""
    experiment_code = ""
    used_all_experiment = []
    all_chart_list = []

    for chart in chart_paths:
        for add_info in experiment_add_info:
            if chart == add_info[0].replace("\\", "/"):
                all_chart_list.append([add_info[0], add_info[1]])
                if add_info[2] not in used_all_experiment:
                    used_all_experiment.append(add_info[2])
                    draw_code += f"\n```\n{add_info[3]}\n```\n"
                    experiment_code += f"\n```\n{experiment_set[add_info[2]]}\n```\n"

    content_parts = [text_part(f"# Draw code:\n{draw_code}\n# Experiment code:\n{experiment_code}")]

    for chart in all_chart_list:
        content_parts.append(image_part(chart[1], "png"))
        content_parts.append(text_part(f"The path of this chart is: {chart[0]}"))

    response = _call_schema(model_name, 1, CHART_CAPTION_SYSTEM, content_parts, CAPTION)
    return response["caption"]



def _replace_chart_captions(model_name: str, tex_code: str, replace_map: dict, tex_template: str) -> str:
    system_prompt = CAPTION_CHART_REPLACE_SYSTEM.replace("STY_FORMAT_DEMAND", tex_template)
    content = (
        f"# Need to replace the LaTeX content in the caption of the chart:\n```\n{tex_code}\n```\n"
        f"# Used for chart paths and captions:\n{json.dumps(replace_map)}"
    )
    response = _call_schema(
        model_name,
        1,
        system_prompt,
        [text_part(content)],
        CAPTION_REPLACE,
    )
    return response["capton_replace_result"]



def integration_writing(task_memory: TaskMemory, experiment_add_info: list, config: dict, experiment_set: dict) -> None:
    with open(config["tex_template_path"], "r", encoding="utf-8") as f:
        tex_template = f.read()

    logging.info("integration start")
    integration_prompt = EXPERIMENT_INTEGRATION_WRITE_SYSTEM.format(
        OVERALL_WRITE_TASK=task_memory.writing_requirement
    )
    integration_init_write_result = _call_schema(
        config["integration_model_name"],
        1,
        integration_prompt,
        [text_part(task_memory.done_write)],
        Integration_Write,
    )["integration_result"]

    generate_tex_prompt = EXPERIMENT_GENERATE_TEX_SYSTEM.replace("STY_FORMAT_DEMAND", tex_template)
    integration_write_result = _call_schema(
        config["integration_model_name"],
        1,
        generate_tex_prompt,
        [text_part(integration_init_write_result)],
        Generate_Tex,
    )["tex_code"]

    try_num = 0
    while try_num < 3:
        divide_chart = _divide_chart(config["text_model_name"], integration_write_result)

        true_chart_list = []
        divide_chart_list = []

        for experiment_add_info_one in experiment_add_info:
            true_chart_list.append(experiment_add_info_one[0].replace("\\", "/"))

        for key in divide_chart:
            if isinstance(divide_chart[key], str):
                divide_chart_list.append(divide_chart[key])
            else:
                for key_key in divide_chart[key]:
                    divide_chart_list.append(divide_chart[key][key_key])

        if Counter(divide_chart_list) == Counter(true_chart_list):
            break
        try_num += 1
        logging.info("divide chart mismatch, retry %s", try_num)

    if try_num >= 3:
        raise RuntimeError("divide chart failed")

    all_need_replace_chart = {}
    for key in divide_chart:
        if isinstance(divide_chart[key], str):
            caption = _generate_caption(
                _get_multimodal_model(config),
                [divide_chart[key]],
                experiment_add_info,
                experiment_set,
            )
            need_place = divide_chart[key]
            all_need_replace_chart[need_place] = caption
        else:
            caption = _generate_caption(
                _get_multimodal_model(config),
                [divide_chart[key][key_key] for key_key in divide_chart[key]],
                experiment_add_info,
                experiment_set,
            )

            need_place = []
            for key_key in divide_chart[key]:
                need_place.append(divide_chart[key][key_key])

            all_need_replace_chart[" and ".join(need_place)] = caption
            need_place = " and ".join(need_place)

        logging.info("caption for %s", need_place)

    integration_write_result = _replace_chart_captions(
        config["text_model_name"],
        integration_write_result,
        all_need_replace_chart,
        tex_template,
    )

    final_tex = _call_schema(
        config["text_model_name"],
        1,
        EXPERIMENT_TYPESETTING_SYSTEM,
        [text_part(integration_write_result)],
        Generate_Tex,
    )["tex_code"]

    os.makedirs(os.path.dirname(config["result_write_path"]), exist_ok=True)
    with open(config["result_write_path"], "w", encoding="utf-8") as f:
        f.write(final_tex)



def write_experiment(config: dict):
    perspectives = os.listdir(config["result_perspective_path"])

    experiment_add_info = []
    for idx, perspective in enumerate(perspectives):
        with open(
            os.path.join(config["result_perspective_path"], perspective, "draw_code.py"),
            "r",
            encoding="utf-8",
        ) as f:
            draw_code = f.read()

        charts_dir = os.path.join(config["result_perspective_path"], perspective, "charts")
        if not os.path.isdir(charts_dir):
            continue

        for chart in os.listdir(charts_dir):
            experiment_add_info.append(
                (
                    os.path.join(f"perspective/{perspective}/charts", chart),
                    encode_image(os.path.join(charts_dir, chart)),
                    f"Experiment {idx+1}",
                    draw_code,
                )
            )

    writing_requirement, experiment_set = get_writing_requirement(config, perspectives)

    temperature = _get_temperature(config)
    outline_content = _build_outline_content(writing_requirement, experiment_add_info)
    outline_response = _call_json(
        _get_multimodal_model(config),
        temperature,
        EXPERIMENT_WRITE_OUTLINE_SYSTEM,
        outline_content,
    )
    if isinstance(outline_response, dict) and "outline" in outline_response:
        task_outline = outline_response["outline"]
    else:
        task_outline = outline_response
    task_memory = TaskMemory(task_outline, writing_requirement)

    logging.info("start writing")
    execute_task(task_memory, experiment_add_info, config)

    logging.info("start integration")
    integration_writing(task_memory, experiment_add_info, config, experiment_set)

    task_memory.log_outline_result(config)



def doing_makedirs(config: dict):
    if not os.path.exists(config["log_path"]):
        os.makedirs(config["log_path"])

    if not os.path.exists(os.path.dirname(config["result_write_path"])):
        os.makedirs(os.path.dirname(config["result_write_path"]))
