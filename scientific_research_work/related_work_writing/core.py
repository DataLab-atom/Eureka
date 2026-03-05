from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import bibtexparser
from langchain_community.vectorstores import Chroma

from scientific_research_work.common.judge_evol.judge.agent_as_a_judge.multimodal_eval_agent import (
    MultimodalJudgeAgent,
)
from llm.embeddings import call_Embedding_model, get_Embedding_model
from llm.client import LLMClient
from llm.message import image_part, system, text_part, user_content
from llm.rerank import reranking_intercept
from scientific_research_work.common.fs import ensure_dir
from scientific_research_work.common.writing_utils import build_writing_requirement, format_reference

from .prompts import (
    INTEGRATE_RELATED_WORK_SYSTEM,
    RELATED_FIELDS_SYSTEM,
    REMOVE_USELESS_BIB_SYSTEM,
    THINK_RELATED_WORK_SYSTEM,
    WRITE_RELATED_WORK_SYSTEM,
)


log_file_name = None


def doing_makedirs(config: dict) -> None:
    ensure_dir(config["log_path"])

    result_dir = os.path.dirname(config["result_write_path"])
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)


def get_research_explain(our_method_path: str, research_information_path: str) -> str:
    return build_writing_requirement(our_method_path, research_information_path)


def selected_cite(
    refer_data: list,
    all_paper_bib_list: set,
    current_refer_data: dict,
    maximum_data_num: int,
    maximum_cite_num: int,
) -> None:
    for data in refer_data:
        if len(current_refer_data.keys()) >= maximum_cite_num:
            break

        if "image_references" in data.metadata:
            bib_tex = data.metadata["image_references"]
            if bib_tex not in all_paper_bib_list:
                if bib_tex not in current_refer_data:
                    current_refer_data[bib_tex] = [data]
                elif len(current_refer_data[bib_tex]) < maximum_data_num:
                    current_refer_data[bib_tex].append(data)
        else:
            bib_tex = data.metadata["citation"]
            if bib_tex not in all_paper_bib_list:
                if bib_tex not in current_refer_data:
                    current_refer_data[bib_tex] = [data]
                elif len(current_refer_data[bib_tex]) < maximum_data_num:
                    current_refer_data[bib_tex].append(data)


def _call_json(model_name: str, temperature: float, system_prompt: str, parts: list[dict]) -> dict:
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, json_mode=True)


def call_related_fields(model_name: str, temperature: float, field_num: int, research_explain: str) -> list[str]:
    response = _call_json(
        model_name,
        temperature,
        RELATED_FIELDS_SYSTEM.format(FIELD_NUM=field_num),
        [text_part(research_explain)],
    )
    return response["related_fields"]


def call_think_related_work(
    model_name: str,
    temperature: float,
    research_field: str,
    research_content: str,
    current_write_content: str,
) -> str:
    if not current_write_content:
        current_write_content = "none"

    text = (
        "Reflect on the content of the next sentence in the \"Related Work\" section of the paper\n\n"
        f"# The existing written results:\n{current_write_content}\n"
        f"# Research field:\n{research_field}\n"
        f"# Research content:\n{research_content}"
    )

    response = _call_json(model_name, temperature, THINK_RELATED_WORK_SYSTEM, [text_part(text)])
    return response["think_related_work"]


def call_write_related_work(
    model_name: str,
    temperature: float,
    tex_template: str,
    think_related_work_content: str,
    research_explain: str,
    related_work_field: str,
    format_text_reference: list,
    format_image_reference: list,
) -> str:
    content_text = (
        f"# Research content:\n{research_explain}\n"
        f"# Related work field:\n{related_work_field}\n"
        f"# Think related work content:\n{think_related_work_content}\n"
        f"# References:\n{json.dumps(format_text_reference, ensure_ascii=False)}"
    )

    parts = [text_part(content_text)]
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

    response = _call_json(
        model_name,
        temperature,
        WRITE_RELATED_WORK_SYSTEM.format(TEX_TEMPLATE=tex_template),
        parts,
    )
    return response["tex_code"]


def call_integrate_related_work(model_name: str, temperature: float, tex_template: str, text_content: str) -> str:
    response = _call_json(
        model_name,
        temperature,
        INTEGRATE_RELATED_WORK_SYSTEM.format(TEX_TEMPLATE=tex_template),
        [text_part(text_content)],
    )
    return response["tex_code"]


def write_by_field(vectordb, research_explain: str, config: dict):
    related_fields = call_related_fields(
        config["text_model_name"],
        config["temperature"],
        config["field_num"],
        research_explain,
    )

    logging.info("%s个领域分别是:\n%s", config["field_num"], related_fields)

    related_work_done_write = []
    all_bib_content = ""
    all_paper_bib_set = set()

    with open(config["tex_template_path"], "r", encoding="utf-8") as f:
        tex_template = f.read()

    for related_work_field in related_fields:
        done_write = ""
        for current_num in range(config["sentence_num"]):
            think_related_work_content = call_think_related_work(
                config["text_model_name"],
                config["temperature"],
                related_work_field,
                research_explain,
                done_write,
            )

            current_refer_data = {}
            idx = 0
            while len(current_refer_data.keys()) < config["used_cite_num"]:
                refer_data = vectordb.similarity_search(think_related_work_content, config["retrieval_data_num"] + idx)
                refer_data = refer_data[idx : config["retrieval_data_num"] + idx]
                refer_data = reranking_intercept(
                    think_related_work_content,
                    refer_data,
                    config["retrieval_data_num"],
                    config["reranker_path"],
                    config["database_device"],
                    config["reranker_model_source"],
                )

                selected_cite(
                    refer_data,
                    all_paper_bib_set,
                    current_refer_data,
                    config["used_data_num"],
                    config["used_cite_num"],
                )

                idx += config["retrieval_data_num"]
                logging.info("已经检索了%d轮", int(idx / config["retrieval_data_num"]))

            format_text_reference, format_image_reference, _ = format_reference(
                [data for bib in current_refer_data for data in current_refer_data[bib]]
            )
            provided_bib = "\n\n".join(list(current_refer_data.keys()))

            try_num = 0
            while try_num < 3:
                related_work_content = call_write_related_work(
                    config["multimodal_model_name"],
                    config["temperature"],
                    tex_template,
                    think_related_work_content,
                    research_explain,
                    related_work_field,
                    format_text_reference,
                    format_image_reference,
                )

                del_bib = []
                for bib in current_refer_data.keys():
                    parser = bibtexparser.bparser.BibTexParser()
                    bib_id = bibtexparser.loads(bib, parser=parser).entries[0]["ID"]

                    if bib_id not in related_work_content:
                        del_bib.append(bib)

                for bib in del_bib:
                    del current_refer_data[bib]

                questions = [
                    "Please carefully check if the writing content references any literature that does not exist in the provided references?"
                ]

                judge_agent = MultimodalJudgeAgent(config["judge_model_name"])
                judge_result = judge_agent.evaluate(
                    f"- The Writing Content:\n```latex\n{related_work_content}\n```\n- The references:\n{json.dumps(format_text_reference, ensure_ascii=False)}",
                    format_image_reference,
                    questions,
                    True,
                )

                if all(judge_result["scores"]):
                    break
                logging.error(
                    "生成写作内容未通过judge, 原因是:%s, 现在开始重试第%d/3次",
                    judge_result["reasons"],
                    try_num + 1,
                )
                try_num += 1

            if try_num >= 3:
                raise RuntimeError("生成写作内容未通过judge, 超出最大重试次数故运行失败")

            used_bib = "\n\n".join(list(current_refer_data.keys()))

            logging.info("关于**%s**第%d句思考内容:\n%s", related_work_field, current_num, think_related_work_content)
            logging.info("关于**%s**第%d句写作内容:\n%s", related_work_field, current_num, related_work_content)
            logging.info("关于**%s**第%d句提供的bib:\n%s", related_work_field, current_num, provided_bib)
            logging.info("关于**%s**第%d句参考的bib:\n%s", related_work_field, current_num, used_bib)

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

            logging.info("关于**%s**第%d句参考的内容:\n%s", related_work_field, current_num, all_refer_content)

            all_bib_content += "\n" + used_bib
            done_write += related_work_content + "\n"

            for bib in current_refer_data.keys():
                all_paper_bib_set.add(bib)

            logging.info("目前使用上的全部bib:\n%s", all_bib_content)

        related_work_done_write.append(
            {
                "related work field": related_work_field,
                "the writing content of the field": done_write,
            }
        )

    logging.info("最终去重前的bib:\n%s", all_bib_content)

    related_work_done_write = call_integrate_related_work(
        config["text_model_name"],
        config["temperature"],
        tex_template,
        json.dumps(related_work_done_write, ensure_ascii=False),
    )

    del_bib = []
    for bib in all_paper_bib_set:
        parser = bibtexparser.bparser.BibTexParser()
        bib_id = bibtexparser.loads(bib, parser=parser).entries[0]["ID"]

        if bib_id not in related_work_done_write:
            del_bib.append(bib)

    for bib in del_bib:
        all_paper_bib_set.remove(bib)

    checked_bib_content = "\n\n".join(list(current_refer_data.keys()))
    logging.info("最终检查后的全部bib:\n%s", checked_bib_content)

    return all_bib_content, all_paper_bib_set, related_work_done_write


def write_related_work(config: dict) -> None:
    if (
        config["embedding_model_source"] != "api"
        and config["reranker_model_source"] != "api"
        and "cpu" not in config["database_device"]
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = config["database_device"][-1]
        config["database_device"] = "cuda"

    if config["embedding_model_source"] == "local":
        embedding_model = get_Embedding_model(config["embedding_model"], config["database_device"])
    else:
        embedding_model = call_Embedding_model(config["embedding_model"])

    vectordb = Chroma(persist_directory=config["persist_directory_path"], embedding_function=embedding_model)

    research_explain = get_research_explain(config["our_method_path"], config["research_information"])

    all_bib_content, all_paper_bib_set, related_work_done_write = write_by_field(vectordb, research_explain, config)

    with open(config["bib_result_path"], "w", encoding="utf-8") as f:
        f.write(all_bib_content)

    with open(config["bib_result_list_path"], "w", encoding="utf-8") as f:
        json.dump(list(all_paper_bib_set), f, indent=2, ensure_ascii=False)

    with open(config["result_write_path"], "w", encoding="utf-8") as f:
        f.write(related_work_done_write)
