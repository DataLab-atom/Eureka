from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import bibtexparser
from langchain_community.vectorstores import Chroma

from llm.embeddings import call_Embedding_model, get_Embedding_model
from llm.client import LLMClient
from llm.message import image_part, system, text_part, user_content
from llm.rerank import reranking_intercept
from scientific_research_work.common.fs import ensure_dir
from scientific_research_work.common.writing_utils import (
    build_writing_requirement,
    check_repeated_cite,
    format_reference,
)

from .prompts import (
    CHECK_REFERENCE_FICTION_SYSTEM,
    DESCRIBE_WRITING_SYSTEM,
    INTEGRATE_INTRODUCTION_SYSTEM,
    INTRODUCTION_WRITE_OUTLINE_SYSTEM,
    REMOVE_USELESS_BIB_SYSTEM,
    WRITE_INTRODUCTION_SYSTEM_1,
    WRITE_INTRODUCTION_SYSTEM_2,
)


class TaskMemory:
    def __init__(self, task: dict, writing_requirement: str):
        self.task = task
        self.task_list = []
        self.now_task_index = 0
        self.all_writing_requirement = writing_requirement
        self.done_write = ""
        self.all_paper_title_list = []
        self.add_all_bib_content = ""
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

    def init_task(self) -> None:
        def init_func(task: dict) -> None:
            space_num = ""
            if task["id"] != "root" and task["id"].count(".") == 0:
                space_num += "\t"
            elif task["id"] == "root":
                pass
            else:
                space_num += (task["id"].count(".") + 1) * "\t"

            self.task_list.append(
                f"{space_num}{task['id']}: {task['goal']}   (task_type: {task['task_type']}, length: {task['length']}, need_cite: {task['need_cite']})"
            )

            if task.get("sub_tasks") and task["sub_tasks"]:
                for plan_one in task["sub_tasks"]:
                    init_func(plan_one)

        init_func(self.task)


def _call_json(model_name: str, temperature: float, system_prompt: str, parts: list[dict]) -> dict:
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, json_mode=True)


def doing_makedirs(config: dict) -> None:
    ensure_dir(config["log_path"])

    result_dir = os.path.dirname(config["result_write_path"])
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)


def call_write_outline(model_name: str, temperature: float, writing_requirement: str) -> dict:
    response = _call_json(model_name, temperature, INTRODUCTION_WRITE_OUTLINE_SYSTEM, [text_part(writing_requirement)])
    return response


def call_think_intro(model_name: str, temperature: float, task: dict, taskmemory: TaskMemory) -> str:
    system_prompt = DESCRIBE_WRITING_SYSTEM.format(
        ALL_WRITE_TASK=taskmemory.all_writing_requirement,
        ALL_WRITE_TASK_PLAN=json.dumps(taskmemory.task),
    )
    response = _call_json(
        model_name,
        temperature,
        system_prompt,
        [text_part(f"# Writing subtasks that require thinking and expression:\n{task['goal']}")],
    )
    return response["think_introduction_work"]


def call_write_intro(
    model_name: str,
    temperature: float,
    system_prompt: str,
    task: dict,
    think_introduction_content: str,
    format_text_reference: list | None = None,
    format_image_reference: list | None = None,
) -> str:
    if format_image_reference is None:
        format_image_reference = []

    base_text = (
        "**All references must be cited in accordance with the requirements of the given LaTeX template where they are needed in the main text, if provided**\n"
        f"# The writing subtask you need to complete is:\n{task['goal']}\n"
        f"# The content of your thinking:\n{think_introduction_content}\n"
        "# References:(**Reiterate, all references must be cited in accordance with the requirements of the given LaTeX template where they are needed in the main text, if provided**)"
    )

    if format_text_reference:
        base_text += "\n" + json.dumps(format_text_reference, ensure_ascii=False)

    parts = [text_part(base_text)]

    for data in format_image_reference:
        parts.append(image_part(data["reference content"]["image_base64"], "jpeg"))
        additional_information = {
            "save path": data["reference content"]["image_path"],
            "Bib citation": data["Bib citation"],
        }
        parts.append(
            text_part(
                f"The relevant information of the reference chart or table:\n{json.dumps(additional_information, ensure_ascii=False)}"
            )
        )

    if (not format_text_reference) and (not format_image_reference):
        parts[0]["text"] += "\nnone"

    response = _call_json(model_name, temperature, system_prompt, parts)
    return response["tex_code"]


def call_check_reference_fiction(model_name: str, temperature: float, bib_content: str, writing_content: str) -> str:
    response = _call_json(
        model_name,
        temperature,
        CHECK_REFERENCE_FICTION_SYSTEM,
        [
            text_part(
                f"# The writing content:\n{writing_content}\n# the existing reference list (BibTex format):\n{bib_content}"
            )
        ],
    )
    return response["result"]


def call_check_reference(model_name: str, temperature: float, bib_content: str, writing_content: str) -> str:
    response = _call_json(
        model_name,
        temperature,
        REMOVE_USELESS_BIB_SYSTEM,
        [text_part(f"# Main text content:{writing_content}\n# All citation information(bib content):\n{bib_content}")],
    )
    return response["tex_code"]


def call_integrate_intro(model_name: str, temperature: float, system_prompt: str, done_write: str) -> str:
    response = _call_json(model_name, temperature, system_prompt, [text_part(done_write)])
    return response["tex_code"]


def write_task(taskmemory: TaskMemory, task: dict, config: dict, vectordb) -> None:
    logging.info("开始执行任务: %s", task["goal"])

    with open(config["tex_template_path"], "r", encoding="utf-8") as f:
        tex_template = f.read()

    if task["need_cite"]:
        think_introduction_content = call_think_intro(
            config["text_model_name"],
            config["temperature"],
            task,
            taskmemory,
        )

        all_refer_data = []
        current_used_paper_title = set()
        idx = 0
        while len(all_refer_data) < config["used_data_num"]:
            refer_data = vectordb.similarity_search(think_introduction_content, config["retrieval_data_num"] + idx)
            refer_data = refer_data[idx : config["retrieval_data_num"] + idx]
            refer_data = reranking_intercept(
                think_introduction_content,
                refer_data,
                config["retrieval_data_num"],
                config["reranker_path"],
                config["database_device"],
                config["reranker_model_source"],
            )

            checked_repeated_cite = check_repeated_cite(
                refer_data,
                taskmemory.all_paper_title_list,
                current_used_paper_title,
            )
            remaining = config["used_data_num"] - len(all_refer_data)
            if len(checked_repeated_cite) > remaining:
                all_refer_data.extend(checked_repeated_cite[:remaining])
            else:
                all_refer_data.extend(checked_repeated_cite)

            idx += config["retrieval_data_num"]
            logging.info("已经检索了%d轮", int(idx / config["retrieval_data_num"]))

        format_text_reference, format_image_reference, bib_content = format_reference(all_refer_data)

        try_num = 1
        while try_num < 4:
            system_prompt = WRITE_INTRODUCTION_SYSTEM_1.format(
                ALL_WRITE_TASK=taskmemory.all_writing_requirement,
                TEX_TEMPLATE=tex_template,
                DONE_WIRTE=taskmemory.done_write,
                ALL_WRITE_TASK_PLAN=json.dumps(taskmemory.task),
            )
            introduction_write_content = call_write_intro(
                config["multimodal_model_name"],
                config["temperature"],
                system_prompt,
                task,
                think_introduction_content,
                format_text_reference,
                format_image_reference,
            )

            if "No" in call_check_reference_fiction(
                config["text_model_name"],
                config["temperature"],
                bib_content,
                introduction_write_content,
            ):
                break

            logging.error("生成写作内容时LLM捏造了参考文献, 现在开始重试第%d/3次", try_num)
            try_num += 1

        if try_num >= 4:
            raise RuntimeError("生成写作内容时LLM捏造了参考文献, 超出最大重试次数故运行失败")

        logging.info("任务:%s 执行结束, 写作结果:\n%s", task["goal"], introduction_write_content)
        logging.info("写作中参考的全部bib:\n%s", bib_content)

        all_refer_content = ""
        for idx, data in enumerate(format_text_reference):
            all_refer_content += (
                f"第{idx}条参考的内容是:\n{data['reference content']}\n"
                f"bib是:\n{data['Bib citation']}"
            )

        all_refer_content += "\n\n下面是参考的图像数据\n\n"

        for idx, data in enumerate(format_image_reference):
            all_refer_content += (
                f"第{idx}条参考的内容是:\n{data['reference content']['image_path']}\n"
                f"bib是:\n{data['Bib citation']}"
            )

        logging.info("任务:%s 参考的内容:\n%s", task["goal"], all_refer_content)

        checked_reference = call_check_reference(
            config["text_model_name"],
            config["temperature"],
            bib_content,
            introduction_write_content,
        )

        logging.info("写作中真正使用了的bib:\n%s", checked_reference)

        taskmemory.done_write += introduction_write_content + "\n\n\n"

        parser = bibtexparser.bparser.BibTexParser()
        check_bib_db = bibtexparser.loads(checked_reference, parser=parser)
        check_paper_list = [entry["title"] for entry in check_bib_db.entries]

        taskmemory.all_paper_title_list.extend(check_paper_list)
        taskmemory.add_all_bib_content += checked_reference + "\n"

        logging.info("介绍章节现已使用的全部bib:\n%s", taskmemory.add_all_bib_content)

    else:
        system_prompt = WRITE_INTRODUCTION_SYSTEM_2.format(
            DONE_WIRTE=taskmemory.done_write,
            ALL_WRITE_TASK=taskmemory.all_writing_requirement,
            ALL_WRITE_TASK_PLAN=json.dumps(taskmemory.task),
        )
        introduction_write_content = call_write_intro(
            config["text_model_name"],
            config["temperature"],
            system_prompt,
            task,
            "",
            None,
            [],
        )
        taskmemory.done_write += introduction_write_content + "\n\n\n"

        logging.info("任务:%s 执行结束, 写作结果:\n%s", task["goal"], introduction_write_content)


def execute_task(taskmemory: TaskMemory, config: dict, vectordb) -> None:
    def execute_step_task(task: dict) -> None:
        if task.get("sub_tasks") and task["sub_tasks"]:
            logging.info("开始执行任务: %s", task["goal"])
            taskmemory.display_task()
            taskmemory.update_task()

            for task_one in task["sub_tasks"]:
                execute_step_task(task_one)

            logging.info("任务:%s 执行结束", task["goal"])
        else:
            taskmemory.display_task()
            taskmemory.update_task()
            write_task(taskmemory, task, config, vectordb)

    execute_step_task(taskmemory.task)


def write_introduction(
    config: dict,
    done_write_content: str | None = None,
    writing_requirement: str | None = None,
) -> None:
    if "cpu" not in config["database_device"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["database_device"][-1]
        config["database_device"] = "cuda"

    if not writing_requirement:
        writing_requirement = build_writing_requirement(config["our_method_path"], config["research_information"])

    if done_write_content is None:
        done_write_content = ""
        for paper_chapter in [
            "related_work_writing/related_work.tex",
            "method_writing/method.tex",
            "experiment_writing/experiment.tex",
        ]:
            chapter_path = os.path.join(os.path.dirname(os.path.dirname(config["result_write_path"])), paper_chapter)
            if os.path.exists(chapter_path):
                with open(chapter_path, "r", encoding="utf-8") as f:
                    done_write_content += f.read() + "\n\n"

    if config["embedding_model_source"] == "local":
        embedding_model = get_Embedding_model(config["embedding_model"], config["database_device"])
    else:
        embedding_model = call_Embedding_model(config["embedding_model"])

    vectordb = Chroma(persist_directory=config["persist_directory_path"], embedding_function=embedding_model)

    outline_response = call_write_outline(
        config["text_model_name"],
        config["temperature"],
        writing_requirement + ("\n" + f"# The content of other chapters in the paper:\n{done_write_content}"),
    )
    task_memory = TaskMemory(
        outline_response["outline"],
        writing_requirement + ("\n" + f"# The content of other chapters in the paper:\n{done_write_content}"),
    )

    if os.path.exists(config["used_reference_bib_list_path"]):
        with open(config["used_reference_bib_list_path"], "r", encoding="utf-8") as f:
            task_memory.all_paper_title_list = json.load(f)

    execute_task(task_memory, config, vectordb)

    with open(config["tex_template_path"], "r", encoding="utf-8") as f:
        tex_template = f.read()

    integrate_prompt = INTEGRATE_INTRODUCTION_SYSTEM.format(
        TEX_TEMPLATE=tex_template,
        ALL_WRITE_TASK=writing_requirement,
        BIB_CONTENT=task_memory.add_all_bib_content,
    )
    integrate_introduction_content = call_integrate_intro(
        config["integration_model_name"],
        config["temperature"],
        integrate_prompt,
        task_memory.done_write,
    )

    logging.info("现有的暂未最终去重的全部bib:\n%s", task_memory.add_all_bib_content)

    task_memory.add_all_bib_content = call_check_reference(
        config["text_model_name"],
        config["temperature"],
        task_memory.add_all_bib_content,
        integrate_introduction_content,
    )

    with open(config["result_write_path"], "w", encoding="utf-8") as f:
        f.write(integrate_introduction_content)

    with open(config["bib_result_path"], "a", encoding="utf-8") as f:
        f.write("\n" + task_memory.add_all_bib_content)

    logging.info("最终检查后介绍章节新增的bib:\n%s", task_memory.add_all_bib_content)

    parser = bibtexparser.bparser.BibTexParser()
    check_bib_db = bibtexparser.loads(task_memory.add_all_bib_content, parser=parser)
    check_paper_list = [entry["title"] for entry in check_bib_db.entries]

    logging.info("最终检查后介绍章节新增的bib论文标题:\n%s", check_paper_list)

    with open(
        os.path.join(os.path.dirname(config["used_reference_bib_list_path"]), "introduction_reference_bib.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(check_paper_list, f, indent=2, ensure_ascii=False)
