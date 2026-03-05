from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from llm.client import LLMClient
from llm.message import image_part, system, text_part, user_content
from scientific_research_work.common.fs import ensure_dir
from scientific_research_work.common.writing_utils import build_writing_requirement

from .prompts import (
    METHOD_GENERATE_TEX_SYSTEM,
    METHOD_INTEGRATION_WRITE_SYSTEM,
    METHOD_THINK_TASK_SYSTEM,
    METHOD_WRITE_OUTLINE_SYSTEM,
    METHOD_WRITE_TASK_SYSTEM,
    METHODOLOGY_IMPROVE_SYSTEM,
    METHODOLOGY_TYPESETTING_SYSTEM,
)


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

    def update_task(self) -> None:
        self.now_task_index += 1

    def display_task(self) -> None:
        need_display_content = ""
        for index, task in enumerate(self.task_list):
            if index == self.now_task_index:
                need_display_content += f"*{task}\n"
            else:
                need_display_content += f"{task}\n"

        logging.info("\n%s", need_display_content)

    def log_outline_result(self, config: dict) -> None:
        outline_result = (
            f"# {'-'*46} Write Outline {'-'*46}\n\n\n{self.task_outline}\n\n\n"
            + f"# {'-'*47} Task Result {'-'*47}\n\n\n"
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
                self.task_list.append(
                    f"{space_num}{task['id']}: {task['goal']}   (task_type: {task['task_type']}, length: {task['length']})"
                )
                self.task_outline += (
                    f"{space_num}**{task['id']}**\n{space_num}: {task['goal']}   (task_type: {task['task_type']}, length: {task['length']})\n\n"
                )
            else:
                self.task_list.append(
                    f"{space_num}{task['id']}: {task['goal']}   (task_type: {task['task_type']})"
                )
                self.task_outline += (
                    f"{space_num}**{task['id']}**\n{space_num}: {task['goal']}   (task_type: {task['task_type']})\n\n"
                )

            if task.get("sub_tasks") and task["sub_tasks"]:
                for plan_one in task["sub_tasks"]:
                    init_func(plan_one)

        init_func(self.task)


def _call_json(model_name: str, temperature: float, system_prompt: str, parts: list[dict]) -> dict:
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, json_mode=True)


def _normalize_outline(writing_requirement: str, outline: object) -> dict:
    if isinstance(outline, dict) and "id" in outline:
        return outline
    if isinstance(outline, dict) and "outline" in outline:
        return _normalize_outline(writing_requirement, outline["outline"])
    if isinstance(outline, list):
        return {
            "id": "root",
            "task_type": "write",
            "goal": "Write the Methodology section based on the provided requirements.",
            "length": "800 words",
            "sub_tasks": outline,
        }
    logging.warning("Unexpected outline format, falling back to empty outline: %s", type(outline))
    return {
        "id": "root",
        "task_type": "write",
        "goal": "Write the Methodology section based on the provided requirements.",
        "length": "800 words",
        "sub_tasks": [],
    }


def _task_parts(task: dict, experiment_add_info: list[tuple] | None) -> list[dict]:
    parts: list[dict] = []
    chart_tables = task.get("need_chart_table") or []

    if chart_tables:
        parts.append(
            text_part(
                f"# The writing subtask you need to complete is:\n {task['goal']}. "
                f"Storage path of experimental result chart:{json.dumps(chart_tables)}"
            )
        )
        parts.append(text_part("# The experimental result chart is as follows:"))
        if experiment_add_info:
            for chart_table in chart_tables:
                for add_info in experiment_add_info:
                    if add_info[0] == chart_table and os.path.splitext(add_info[0])[1] == ".png":
                        parts.append(image_part(add_info[1], "png"))
                        the_chart_info = {
                            "Draw code": add_info[3],
                            "Save path": add_info[0],
                            "From experiment": add_info[2],
                        }
                        parts.append(
                            text_part(
                                f"Information in the chart of the results of this experiment:{json.dumps(the_chart_info)}"
                            )
                        )
    else:
        parts.append(text_part(f"# The writing subtask you need to complete is:\n {task['goal']}"))

    return parts


def call_write_outline(model_name: str, temperature: float, writing_requirement: str, experiment_add_info=None) -> dict:
    parts = [text_part(writing_requirement)]
    if experiment_add_info:
        parts.append(text_part("# The all experimental charts of results is as follows:"))
        for experiment_add_info_one in experiment_add_info:
            if os.path.splitext(experiment_add_info_one[0])[1] == ".png":
                parts.append(image_part(experiment_add_info_one[1], "png"))
                chart_info = {
                    "Draw code": experiment_add_info_one[3],
                    "Save path": experiment_add_info_one[0],
                    "From experiment": experiment_add_info_one[2],
                }
                parts.append(
                    text_part(
                        f"Information in the chart of the result of the experiment:\n{json.dumps(chart_info)}"
                    )
                )

    return _call_json(model_name, temperature, METHOD_WRITE_OUTLINE_SYSTEM, parts)


def call_think_task(model_name: str, temperature: float, task: dict, related_think: str, memory: TaskMemory) -> str:
    system_prompt = METHOD_THINK_TASK_SYSTEM.format(
        ALL_WRITE_TASK=memory.writing_requirement,
        ALL_WRITE_TASK_PLAN=json.dumps(memory.task),
        RELATED_THINK=related_think,
        DONE_WRITE=memory.done_write,
    )
    response = _call_json(model_name, temperature, system_prompt, _task_parts(task, None))
    return response["think"]


def call_write_task(model_name: str, temperature: float, task: dict, related_think: str, memory: TaskMemory) -> str:
    system_prompt = METHOD_WRITE_TASK_SYSTEM.format(
        ALL_WRITE_TASK=memory.writing_requirement,
        ALL_WRITE_TASK_PLAN=json.dumps(memory.task),
        RELATED_THINK=related_think,
        DONE_WRITE=memory.done_write,
    )
    response = _call_json(model_name, temperature, system_prompt, _task_parts(task, None))
    return response["write"]


def call_integration(model_name: str, temperature: float, writing_requirement: str, done_write: str) -> str:
    system_prompt = METHOD_INTEGRATION_WRITE_SYSTEM.format(OVERALL_WRITE_TASK=writing_requirement)
    response = _call_json(model_name, temperature, system_prompt, [text_part(done_write)])
    return response["integration_result"]


def call_improve(model_name: str, temperature: float, writing_requirement: str, content: str) -> str:
    system_prompt = METHODOLOGY_IMPROVE_SYSTEM.format(OVERALL_WRITE_TASK=writing_requirement)
    response = _call_json(model_name, temperature, system_prompt, [text_part(content)])
    return response["methodology_improve_content"]


def call_generate_tex(model_name: str, temperature: float, tex_template: str, content: str) -> str:
    system_prompt = METHOD_GENERATE_TEX_SYSTEM.replace("STY_FORMAT_DEMAND", tex_template)
    response = _call_json(model_name, temperature, system_prompt, [text_part(content)])
    return response["tex_code"]


def call_typeset(model_name: str, temperature: float, tex_template: str, tex_code: str) -> str:
    system_prompt = METHODOLOGY_TYPESETTING_SYSTEM.replace("STY_FORMAT_DEMAND", tex_template)
    response = _call_json(model_name, temperature, system_prompt, [text_part(tex_code)])
    return response["tex_code"]


def write_task(taskmemory: TaskMemory, task: dict, related_think: str, config: dict) -> None:
    logging.info("开始执行任务: %s", task["goal"])

    taskmemory.display_task()
    taskmemory.update_task()

    write_content = call_write_task(
        config["write_model_name"],
        config["temperature"],
        task,
        related_think,
        taskmemory,
    )

    taskmemory.done_write += write_content + "\n\n\n"

    logging.info("任务:%s 执行结束, 写作结果:\n%s\n", task["goal"], write_content)
    taskmemory.done_write_dict[task["goal"]] = write_content


def think_task(taskmemory: TaskMemory, task: dict, related_think: str, config: dict) -> str:
    logging.info("开始执行任务: %s", task["goal"])

    taskmemory.display_task()
    taskmemory.update_task()

    think_result = call_think_task(
        config["write_model_name"],
        config["temperature"],
        task,
        related_think,
        taskmemory,
    )

    logging.info("任务:%s 执行结束, 思考结果:\n%s\n", task["goal"], think_result)
    taskmemory.done_write_dict[task["goal"]] = think_result

    return think_result


def execute_task(taskmemory: TaskMemory, config: dict) -> None:
    def execute_step_task(task: dict, related_think: str) -> str:
        this_layer_related_think = ""

        if task.get("sub_tasks") and task["sub_tasks"]:
            logging.info("开始执行任务: %s", task["goal"])
            taskmemory.display_task()
            taskmemory.update_task()

            for task_one in task["sub_tasks"]:
                result_task = execute_step_task(task_one, related_think + this_layer_related_think)
                if result_task:
                    this_layer_related_think += result_task + "\n\n\n"

            logging.info("任务:%s 执行结束", task["goal"])
        else:
            if task["task_type"] == "think":
                return think_task(taskmemory, task, related_think, config)
            if task["task_type"] == "write":
                write_task(taskmemory, task, related_think, config)

        return ""

    execute_step_task(taskmemory.task, "")


def integration_writing(done_write: str, writing_requirement: str, config: dict) -> None:
    logging.info("现有未整理的全部markdown结果:\n%s", done_write)

    integration_init_write_result = call_integration(
        config["integration_model_name"],
        config["temperature"],
        writing_requirement,
        done_write,
    )

    logging.info("初步整理后的markdown结果:\n%s", integration_init_write_result)

    improve_integration = integration_init_write_result
    for idx in range(config["improve_round"]):
        improve_integration = call_improve(
            config["write_model_name"],
            config["temperature"],
            writing_requirement,
            integration_init_write_result,
        )

        if ("already completed" in improve_integration) or (config["improve_round"] - 1 == idx):
            improve_integration = integration_init_write_result
            break
        integration_init_write_result = improve_integration
        logging.info("第%s轮优化结果:\n%s", idx + 1, integration_init_write_result)

    logging.info("最终优化结果:\n%s", improve_integration)

    with open(config["tex_template_path"], "r", encoding="utf-8") as f:
        tex_template = f.read()

    current_tex = call_generate_tex(
        config["write_model_name"],
        config["temperature"],
        tex_template,
        improve_integration,
    )

    logging.info("Latex代码:\n%s", current_tex)

    final_tex = call_typeset(
        config["write_model_name"],
        config["temperature"],
        tex_template,
        current_tex,
    )

    with open(config["result_write_path"], "w", encoding="utf-8") as f:
        f.write(final_tex)


def write_methodology(config: dict, writing_requirement: str | None = None) -> None:
    if not writing_requirement:
        writing_requirement = build_writing_requirement(config["our_method_path"], config["research_information"])

    outline_response = call_write_outline(
        config["write_model_name"],
        config["temperature"],
        writing_requirement,
    )
    outline = outline_response.get("outline", outline_response)
    task_memory = TaskMemory(_normalize_outline(writing_requirement, outline), writing_requirement)

    logging.info("开始按写作规划写方法论部分")
    execute_task(task_memory, config)

    logging.info("开始整合写作方法论部分")
    integration_writing(task_memory.done_write, writing_requirement, config)

    task_memory.log_outline_result(config)

    logging.info("方法论章节写作结束")


def doing_makedirs(config: dict) -> None:
    ensure_dir(config["log_path"])
    ensure_dir(os.path.dirname(config["result_write_path"]))
