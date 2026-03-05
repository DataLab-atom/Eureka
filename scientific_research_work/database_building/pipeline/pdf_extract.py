import copy
import logging
import os
import time
import zipfile
from pathlib import Path

import requests
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode

from llm.env import get_env
from .md_postprocess import data_dealing


def _process_output(
    pdf_info,
    pdf_bytes,
    pdf_file_name,
    local_md_dir,
    local_image_dir,
    md_writer,
    f_draw_layout_bbox,
    f_draw_span_bbox,
    f_dump_orig_pdf,
    f_dump_md,
    f_dump_content_list,
    f_dump_middle_json,
    f_dump_model_output,
    f_make_md_mode,
    middle_json,
    model_output=None,
    is_pipeline=True,
) -> None:
    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )


def do_parse(
    output_dir: str,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    p_lang_list: list[str],
    backend: str = "pipeline",
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
    server_url: str | None = None,
    f_draw_layout_bbox: bool = True,
    f_draw_span_bbox: bool = True,
    f_dump_md: bool = True,
    f_dump_middle_json: bool = True,
    f_dump_model_output: bool = True,
    f_dump_orig_pdf: bool = True,
    f_dump_content_list: bool = True,
    f_make_md_mode: MakeMode = MakeMode.MM_MD,
    start_page_id: int = 0,
    end_page_id: int | None = None,
) -> None:
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        pdf_bytes_list[idx] = new_pdf_bytes

    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
        pdf_bytes_list,
        p_lang_list,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )

    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir = os.path.join(output_dir, pdf_file_name, "images")
        local_md_dir = os.path.join(output_dir, pdf_file_name)
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            _lang,
            _ocr_enable,
            formula_enable,
        )

        pdf_info = middle_json["pdf_info"]
        pdf_bytes = pdf_bytes_list[idx]
        _process_output(
            pdf_info,
            pdf_bytes,
            pdf_file_name,
            local_md_dir,
            local_image_dir,
            md_writer,
            f_draw_layout_bbox,
            f_draw_span_bbox,
            f_dump_orig_pdf,
            f_dump_md,
            f_dump_content_list,
            f_dump_middle_json,
            f_dump_model_output,
            f_make_md_mode,
            middle_json,
            model_json,
            is_pipeline=True,
        )


def _resolve_pdf_md_token(pdf_md_token: str | None) -> str:
    token = pdf_md_token or get_env("PDF_MD_TOKEN", "")
    if not token:
        raise ValueError("PDF_MD_TOKEN is not set")
    return token


def load_extractor_pdf(pdf_data_path_list: list, save_result_path: str, pdf_md_token: str | None) -> None:
    token = _resolve_pdf_md_token(pdf_md_token)
    url = "https://mineru.net/api/v4/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = {
        "language": "en",
        "files": [
            {"name": f"paper_{idx:03}.pdf", "data_id": f"paper_{idx:03}"}
            for idx in range(len(pdf_data_path_list))
        ],
    }

    try:
        response = requests.post(url, headers=header, json=data)
        if response.status_code == 200:
            result = response.json()
            logging.info("response success. result:%s", result)
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                logging.info("batch_id:%s,urls:%s", batch_id, urls)
                for i in range(0, len(urls)):
                    with open(pdf_data_path_list[i], "rb") as f:
                        res_upload = requests.put(urls[i], data=f)
                        if res_upload.status_code == 200:
                            logging.info("%s upload success", urls[i])
                        else:
                            logging.error("%s upload failed", urls[i])
                            raise RuntimeError("upload failed")
            else:
                raise RuntimeError("apply upload url failed")
        else:
            raise RuntimeError("response not success")
    except Exception as err:
        logging.error(err)
        raise

    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    done_flag = False
    while not done_flag:
        res = requests.get(url, headers=header)
        if all(state == "done" for state in [data["state"] for data in res.json()["data"]["extract_result"]]):
            break
        time.sleep(15)

    for idx in range(len(pdf_data_path_list)):
        folder_name = ".".join(os.path.basename(pdf_data_path_list[idx]).split(".")[:-1])
        download_url = res.json()["data"]["extract_result"][idx]["full_zip_url"]
        response = requests.get(download_url, proxies={"http": None, "https": None})
        zip_filename = url.split("/")[-1]
        os.makedirs(os.path.join(save_result_path, folder_name), exist_ok=True)
        with open(os.path.join(save_result_path, folder_name, zip_filename), "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(os.path.join(save_result_path, folder_name, zip_filename), "r") as zip_ref:
            zip_ref.extractall(os.path.join(save_result_path, folder_name))
        os.remove(os.path.join(save_result_path, folder_name, zip_filename))

        for name in os.listdir(os.path.join(save_result_path, folder_name)):
            if name == "images":
                continue
            if name.split(".")[1] != "md":
                os.remove(os.path.join(save_result_path, folder_name, name))
            else:
                os.rename(
                    os.path.join(save_result_path, folder_name, name),
                    os.path.join(save_result_path, folder_name, f"{folder_name}.md"),
                )


def extractor_zip(pdf_data_path_list: list, save_result_path: str) -> None:
    for idx in range(len(pdf_data_path_list)):
        folder_name = ".".join(os.path.basename(pdf_data_path_list[idx]).split(".")[:-1])
        os.makedirs(os.path.join(save_result_path, folder_name), exist_ok=True)

        with zipfile.ZipFile(pdf_data_path_list[idx], "r") as zip_ref:
            zip_ref.extractall(os.path.join(save_result_path, folder_name))
        os.remove(pdf_data_path_list[idx])

        for name in os.listdir(os.path.join(save_result_path, folder_name)):
            if name == "images":
                continue
            elif name.split(".")[1] == "pdf":
                os.rename(
                    os.path.join(save_result_path, folder_name, name),
                    os.path.join(os.path.dirname(pdf_data_path_list[idx]), f"{folder_name}.pdf"),
                )
            elif name.split(".")[1] == "md":
                os.rename(
                    os.path.join(save_result_path, folder_name, name),
                    os.path.join(save_result_path, folder_name, f"{folder_name}.md"),
                )
            else:
                os.remove(os.path.join(save_result_path, folder_name, name))


def pdf_extractor(
    pdf_data_path_list: list,
    save_result_path: str,
    pdf_transform_mode: str,
    api_parsing_method: str,
    pdf_md_token: str | None = None,
) -> None:
    if pdf_transform_mode == "api" and api_parsing_method == "file":
        load_extractor_pdf(pdf_data_path_list, save_result_path, pdf_md_token)
    elif pdf_transform_mode == "api" and api_parsing_method == "url":
        extractor_zip(pdf_data_path_list, save_result_path)
    else:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []

        for pdf_data_path in pdf_data_path_list:
            file_name = str(Path(pdf_data_path).stem)
            pdf_bytes = read_fn(pdf_data_path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append("en")

        do_parse(
            output_dir=save_result_path,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend="pipeline",
            parse_method="auto",
            server_url=None,
            start_page_id=0,
            end_page_id=None,
        )

    for folder in os.listdir(save_result_path):
        if not os.path.exists(os.path.join(save_result_path, folder, "images")):
            os.makedirs(os.path.join(save_result_path, folder, "images"))

    data_dealing(save_result_path, len(os.listdir(save_result_path)) - len(pdf_data_path_list))
