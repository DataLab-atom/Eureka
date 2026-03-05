import json
import logging
import os
import random

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from llm.embeddings import call_Embedding_model, get_Embedding_model
from llm.rerank import reranking_intercept
from llm.client import LLMClient
from llm.message import image_part, system, text_part, user_content
from scientific_research_work.experiment.prompts.experimenting import (
    ANALYSIS_ANGLE,
    ANALYSIS_ANGLE_SYSTEM,
    EXPERIMENTAL_INDOCATORS,
    EXPERIMENTAL_INDOCATORS_SYSTEM,
    INFORMATION_ARCHITECTURE,
    INFORMATION_ARCHITECTURE_SYSTEM,
    MAIN_RESEARCH,
    MAIN_RESEARCH_SYSTEM,
    METHOD_NAMING,
    METHOD_NAMING_SYSTEM,
    METHOD_STATEMENT,
    METHOD_STATEMENT_SYSTEM,
    PAPER_ABSTRACT_EXTRACT,
    PAPER_ABSTRACT_EXTRACT_SYSTEM,
)

random.seed(2025)


def text_image_separate(relevant_background_knowledge: list[Document]):
    text_data = []
    image_data = []

    for data in relevant_background_knowledge:
        if data.metadata.get("image_base64"):
            image_data.append(data)
        else:
            text_data.append(data)

    return text_data, image_data


def _call_schema(model_name: str, temperature: float, system_prompt: str, parts: list[dict], response_schema):
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, response_schema=response_schema)


def _paper_abstract_extract(model_name: str, markdown_content: str) -> str:
    response = _call_schema(
        model_name,
        0.5,
        PAPER_ABSTRACT_EXTRACT_SYSTEM,
        [text_part(markdown_content)],
        PAPER_ABSTRACT_EXTRACT,
    )
    return response["abstract"]


def _main_research(model_name: str, summary: str) -> str:
    response = _call_schema(
        model_name,
        1,
        MAIN_RESEARCH_SYSTEM,
        [text_part(summary)],
        MAIN_RESEARCH,
    )
    return response["main_research"]


def _method_naming(model_name: str, content: str) -> str:
    response = _call_schema(
        model_name,
        1,
        METHOD_NAMING_SYSTEM,
        [text_part(content)],
        METHOD_NAMING,
    )
    return response["method_naming"]


def _method_statement(model_name: str, python_code: str) -> str:
    response = _call_schema(
        model_name,
        0.3,
        METHOD_STATEMENT_SYSTEM,
        [text_part(python_code)],
        METHOD_STATEMENT,
    )
    return response["method_statement"]


def _information_architecture(model_name: str, method_explain: str) -> str:
    response = _call_schema(
        model_name,
        1,
        INFORMATION_ARCHITECTURE_SYSTEM,
        [text_part(method_explain)],
        INFORMATION_ARCHITECTURE,
    )
    return response["information_architecture"]


def _analysis_perspective(
    model_name: str,
    method_explain: str,
    all_analysis_angle: dict,
    database_chance: float,
    text_data_list: list[Document],
    image_data_list: list[Document],
) -> dict:
    if not all_analysis_angle:
        all_analysis_angle = "Not have"

    if database_chance < 0.5:
        text_content = "Not have"
    else:
        text_content = ""
        for text_data in text_data_list:
            text_content += f"{json.dumps(text_data.metadata)}\n\n"

    content_parts = [
        text_part(
            f"{method_explain}\n# Existing analysis angle:\n{all_analysis_angle}\n# Relevant text and image background knowledge:\n{text_content}"
        )
    ]

    if database_chance >= 0.5:
        for image_content in image_data_list:
            content_parts.append(image_part(image_content.metadata["image_base64"], "jpeg"))
            tmp_dict = {
                "the image path": image_content.metadata["image_path"],
                "the image references": image_content.metadata["image_references"],
            }
            content_parts.append(text_part(f"The information of the relevant image background knowledge:\n{tmp_dict}"))

    response = _call_schema(model_name, 1, ANALYSIS_ANGLE_SYSTEM, content_parts, ANALYSIS_ANGLE)

    required_keys = [
        "analysis_perspective",
        "core_focus",
        "differentiated_feature",
        "expected_insight",
        "experimental_metric",
    ]
    if not all(key in response for key in required_keys):
        raise RuntimeError("analysis perspective response missing required keys")

    return response


def _experimental_indicators(
    model_name: str,
    research_content: str,
    text_data_list: list[Document],
    image_data_list: list[Document],
) -> str:
    content_parts = [text_part(f"{research_content}\n# Refer to relevant materials:\n")]

    for data in text_data_list:
        content_parts.append(text_part(f"\n{json.dumps(data.metadata)}\n"))

    for data in image_data_list:
        content_parts.append(image_part(data.metadata["image_base64"], "png"))
        additional_information = {
            "save path": data.metadata["image_path"],
            "reference": data.metadata["image_references"],
        }
        content_parts.append(
            text_part(f"The relevant information of the reference chart or table:\n{json.dumps(additional_information)}")
        )

    if not text_data_list and not image_data_list:
        content_parts[0]["text"] += "none"

    response = _call_schema(
        model_name,
        1,
        EXPERIMENTAL_INDOCATORS_SYSTEM,
        content_parts,
        EXPERIMENTAL_INDOCATORS,
    )
    return response["suitable_drawing_chart"]


def get_method_exlain(config: dict):
    method_exlain = None
    method_statement = None
    method_statement_flag = False

    with open(os.path.join(config["method_code_path"], "Ours.py"), "r", encoding="utf-8") as f:
        ours_code = f.read()

    if not os.path.exists(config["reference_information"]):
        reference_information = []
        with open(config["bib_path"], "r", encoding="utf-8") as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        markdown_folder_list = sorted(os.listdir(config["save_md_path"]))

        for idx, entry in enumerate(bib_database.entries):
            with open(
                os.path.join(config["save_md_path"], markdown_folder_list[idx], f"{markdown_folder_list[idx]}.md"),
                "r",
                encoding="utf-8",
            ) as f:
                lines = f.readlines()[:30]
            abstract = ""
            for line in lines:
                abstract += line

            db = BibDatabase()
            db.entries = [entry]
            writer = BibTexWriter()

            abstract_text = _paper_abstract_extract(config["text_model_name"], abstract)
            reference_information.append({"abstract": abstract_text, "bib_tex": writer.write(db)})

        with open(config["reference_information"], "w", encoding="utf-8") as f:
            json.dump(reference_information, f, ensure_ascii=False, indent=4)
    else:
        with open(config["reference_information"], "r", encoding="utf-8") as f:
            reference_information = json.load(f)

    if not os.path.exists(config["research_information"]):
        main_research = _main_research(config["text_model_name"], json.dumps(reference_information))

        method_name = _method_naming(
            config["text_model_name"],
            f"# {main_research}.For this, we propose a method, the python code implementation of our method is as follows:\n```\n{ours_code}\n```\n",
        )

        with open(config["research_information"], "w", encoding="utf-8") as f:
            json.dump({"method_name": method_name, "main_research": main_research}, f, ensure_ascii=False, indent=4)
    else:
        with open(config["research_information"], "r", encoding="utf-8") as f:
            research_information = json.load(f)

        method_name = research_information["method_name"]
        main_research = research_information["main_research"]

    method_exlain = (
        f"# {main_research}.For this, we have proposed a method:**{method_name}**, the python code implementation of our method is as follows:\n```\n{ours_code}\n```\n# The Python implementation code for the baseline method is as follows:\n"
    )

    if os.path.exists(config["method_statement"]):
        with open(config["method_statement"], "r", encoding="utf-8") as f:
            method_statement = f.read()
            method_statement_flag = True
    else:
        method_statement = f"# Our proposed method:\n{_method_statement(config['text_model_name'], ours_code)}\n# Baseline method:\n"

    method_folder_name = sorted(os.listdir(os.path.join(config["method_code_path"], "other")))
    for folder_name in method_folder_name:
        method_name = os.listdir(os.path.join(config["method_code_path"], "other", folder_name))[0]

        with open(
            os.path.join(config["method_code_path"], "other", folder_name, method_name), "r", encoding="utf-8"
        ) as f:
            code_content = f.read()

        method_exlain += f"\n- {method_name.split('.')[0]} method:\n```\n{code_content}\n```\n"

        if not method_statement_flag:
            method_statement += f"\n- {method_name.split('.')[0]} method\n```\n{_method_statement(config['text_model_name'], code_content)}\n```"

    if not method_statement_flag:
        with open(config["method_statement"], "w", encoding="utf-8") as f:
            f.write(method_statement)

    return method_exlain, method_statement


def get_experimental_indicators(
    method_statement: str,
    method_explain: str,
    vectordb: Chroma,
    config: dict,
    analysis_perspective: dict,
    analysis_perspective_num: int,
):
    analysis_perspective_structured = f"""
    # Analysis perspective: {analysis_perspective['analysis_perspective']}
        - Core focus: {analysis_perspective['core_focus']}
        - Differentiated feature: {analysis_perspective['differentiated_feature']}
        - Expected insight: {analysis_perspective['expected_insight']}
        - Experimental metric:{analysis_perspective['experimental_metric']}
    """.strip()

    draw_tendency_chance = random.uniform(0, 1)
    used_reference_chance = random.uniform(0, 1)

    query = f"{analysis_perspective_structured}\n\n{method_statement}"

    refer_data = vectordb.similarity_search(query, config["retrieval_data_num"])

    relevant_background_knowledge = reranking_intercept(
        query,
        refer_data,
        config["reranked_data_num"],
        config["reranker_path"],
        config["database_device"],
        config["reranker_model_source"],
    )

    text_data_list, image_data_list = text_image_separate(relevant_background_knowledge)

    if draw_tendency_chance < 0.7:
        if used_reference_chance < 0.5:
            visualization_strategy = _experimental_indicators(
                config["multimodel_model_name"],
                f"{method_explain}\n\n{analysis_perspective_structured}\n# If conditions permit, try not to draw line charts as much as possible",
                text_data_list,
                image_data_list,
            )
        else:
            visualization_strategy = _experimental_indicators(
                config["multimodel_model_name"],
                f"{method_explain}\n\n{analysis_perspective_structured}\n# If conditions permit, try not to draw line charts as much as possible",
                [],
                [],
            )
    else:
        if used_reference_chance < 0.5:
            visualization_strategy = _experimental_indicators(
                config["multimodel_model_name"],
                f"{method_explain}\n\n{analysis_perspective_structured}",
                text_data_list,
                image_data_list,
            )
        else:
            visualization_strategy = _experimental_indicators(
                config["multimodel_model_name"],
                f"{method_explain}\n\n{analysis_perspective_structured}",
                [],
                [],
            )

    return visualization_strategy, used_reference_chance < 0.5, draw_tendency_chance < 0.7


def generate_analysis_perspectives(config: dict):
    method_explain, method_statement = get_method_exlain(config)

    information_architecture_content = _information_architecture(config["text_model_name"], method_explain)

    if config["embedding_model_source"] == "local":
        embedding_model = get_Embedding_model(config["embedding_model"], config["database_device"])
    else:
        embedding_model = call_Embedding_model(config["embedding_model"])

    vectordb = Chroma(
        persist_directory=config["experiment_persist_directory_path"],
        embedding_function=embedding_model,
    )

    refer_data = vectordb.similarity_search(information_architecture_content, config["retrieval_data_num"])

    relevant_background_knowledge = reranking_intercept(
        information_architecture_content,
        refer_data,
        config["reranked_data_num"],
        config["reranker_path"],
        config["database_device"],
        config["reranker_model_source"],
    )

    text_data_list, image_data_list = text_image_separate(relevant_background_knowledge)

    analysis_path = config.get("analysis_perspective_path") or os.path.join(
        os.path.dirname(config["reference_data"]), "analysis_perspective_explain.json"
    )

    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            all_analysis_perspective = json.load(f)
    else:
        all_analysis_perspective = {}

    for idx in range(
        len(all_analysis_perspective),
        len(all_analysis_perspective) + config["all_analysis_perspective_num"],
    ):
        database_chance = random.uniform(0, 1)

        analysis_perspective = _analysis_perspective(
            config["multimodel_model_name"],
            method_explain,
            all_analysis_perspective,
            database_chance,
            text_data_list,
            image_data_list,
        )

        all_analysis_perspective[f"analysis_perspective_{idx:03d}"] = {
            "analysis_perspective": analysis_perspective["analysis_perspective"],
            "core_focus": analysis_perspective["core_focus"],
            "differentiated_feature": analysis_perspective["differentiated_feature"],
            "expected_insight": analysis_perspective["expected_insight"],
            "experimental_metric": analysis_perspective["experimental_metric"],
        }

        all_analysis_perspective[f"analysis_perspective_{idx:03d}"]["other_is_refer"] = database_chance >= 0.5

        suitable_drawing_chart, visualization_is_refer, no_line_tendency = get_experimental_indicators(
            method_statement, method_explain, vectordb, config, analysis_perspective, idx
        )
        all_analysis_perspective[f"analysis_perspective_{idx:03d}"][
            "suitable_drawing_chart"
        ] = suitable_drawing_chart
        all_analysis_perspective[f"analysis_perspective_{idx:03d}"][
            "visualization_is_refer"
        ] = visualization_is_refer
        all_analysis_perspective[f"analysis_perspective_{idx:03d}"]["no_line_tendency"] = no_line_tendency

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(all_analysis_perspective, f, indent=4, ensure_ascii=False)

    logging.info("analysis perspective generation finished")

    return method_explain, method_statement, all_analysis_perspective


get_method_explain = get_method_exlain
