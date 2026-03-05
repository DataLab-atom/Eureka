import json
import logging
import os
from functools import partial
from multiprocessing import Pool 
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from scientific_research_work.common.image import encode_image
from scientific_research_work.common.logging import setup_logging
from llm import LLMClient, LLM_PRESETS, system, user_image_base64
from llm.embeddings import call_Embedding_model, get_Embedding_model
from ..prompts import CHART_DESCRIPTION_SYSTEM, ChartDescription
from ..text.split import split_text_with_headings


def get_image_description(image_path: str, model_name: str, image_references: str | None = None) -> tuple:
    image_base = encode_image(image_path)
    client = LLMClient()
    messages = [
        system(CHART_DESCRIPTION_SYSTEM),
        user_image_base64(image_base, "jpeg"),
    ]
    response = client.call(
        messages,
        preset=LLM_PRESETS.CHART_DESCRIPTION,
        model=model_name,
        response_schema=ChartDescription,
    )
    image_description = response["chart_description"]
    return (image_base, image_description, image_path, image_references)


def data_init_processing(
    image_folder_path_list: list[str],
    paper_cite_information: dict,
    config: dict,
    log_file_name: str | None = None,
) -> list[Document]:
    data_list: list[Document] = []
    for image_folder_path in image_folder_path_list:
        folder_name = os.path.basename(os.path.dirname(image_folder_path))
        dir_path = os.path.dirname(image_folder_path)
        cite = paper_cite_information[f"{folder_name}.pdf"]

        with open(os.path.join(dir_path, f"{folder_name}.md"), "r", encoding="utf-8") as f:
            md_content = f.read()
            data_list.extend(
                split_text_with_headings(
                    md_content,
                    cite,
                    config["text_data_processing"]["min_tokens"],
                    config["text_data_processing"]["max_tokens"],
                    config["text_data_processing"]["chunk_size"],
                    config["embedding_model"],
                )
            )

        need_image_description = []
        for image_file in os.listdir(image_folder_path):
            need_image_description.append(os.path.join(image_folder_path, image_file))

        partial_get_image_description = partial(
            get_image_description,
            model_name=config["chart_model_name"],
            image_references=cite,
        )

        if log_file_name:
            pool_kwargs = {"initializer": setup_logging, "initargs": (log_file_name,)}
        else:
            pool_kwargs = {}

        with Pool(processes=5, **pool_kwargs) as pool:
            image_description_results = pool.map(partial_get_image_description, need_image_description)

        for image_description_result in image_description_results:
            data_list.append(
                Document(
                    page_content=image_description_result[1],
                    metadata={
                        "image_base64": image_description_result[0],
                        "image_path": image_description_result[2],
                        "image_references": image_description_result[3],
                    },
                )
            )

    return data_list


def build_paper_vb(
    pdf_path_list: list[str],
    paper_cite_information: dict,
    persist_directory_path: str,
    config: dict,
    log_file_name: str | None = None,
):
    folder_path_list = []
    for pdf_path in pdf_path_list:
        folder_path_list.append(os.path.join(config["save_md_path"], str(Path(pdf_path).stem), "images"))

    vectordb_data_list = data_init_processing(folder_path_list, paper_cite_information, config, log_file_name)

    if config["embedding_model_source"] == "local":
        embedding_model = get_Embedding_model(config["embedding_model"], config["database_device"])
    else:
        embedding_model = call_Embedding_model(config["embedding_model"])

    print(persist_directory_path)


    vectordb = Chroma.from_documents(
        documents=vectordb_data_list,
        embedding=embedding_model,
        persist_directory=persist_directory_path,
    )

    return vectordb


def build_plotcode_vb(config: dict):
    vectordb_data_list = []

    for template in os.listdir(config["plot_code_template"]):
        with open(
            os.path.join(config["plot_code_template"], template, "plot_code.py"),
            "r",
            encoding="utf-8",
        ) as f:
            plot_code = f.read()

        with open(
            os.path.join(config["plot_code_template"], template, "caption_explain.md"),
            "r",
            encoding="utf-8",
        ) as f:
            caption_explain = f.read()

        charts = []
        for chart in os.listdir(os.path.join(config["plot_code_template"], template, "charts")):
            charts.append(os.path.join(config["plot_code_template"], template, "charts", chart))

        vectordb_data_list.append(
            Document(
                page_content=caption_explain,
                metadata={
                    "plot_code": plot_code,
                    "charts": json.dumps(charts),
                },
            )
        )

    if config["embedding_model_source"] == "local":
        embedding_model = get_Embedding_model(config["embedding_model"], config["database_device"])
    else:
        embedding_model = call_Embedding_model(config["embedding_model"])

    vectordb = Chroma.from_documents(
        documents=vectordb_data_list,
        embedding=embedding_model,
        persist_directory=config["PlotCode_vb_path"],
    )

    logging.info("plot code vector database created at %s", config["PlotCode_vb_path"])
    return vectordb
