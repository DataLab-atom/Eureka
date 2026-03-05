import datetime
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from semanticscholar import SemanticScholar

from .pipeline.pdf_extract import pdf_extractor
from .scholar.semantic import check_reference_repeat
from scientific_research_work.common.settings import load_database_config
from .vector_db.build import build_paper_vb, build_plotcode_vb
from .vector_db.stats import count_vector_database
from scientific_research_work.common.paper_crawler.main import paper_crawler
from scientific_research_work.common.fs import clear_folder


def _prepare_directories(config: dict) -> None:
    os.makedirs(config["log_path"], exist_ok=True)

    clear_folder(config["save_pdf_path"])
    clear_folder(config["save_md_path"])
    clear_folder(config["experiment_persist_directory_path"])

    if os.path.exists(config["full_text_persist_directory_path"]):
        shutil.rmtree(config["full_text_persist_directory_path"])
    os.makedirs(config["full_text_persist_directory_path"], exist_ok=True)


def _init_logging(log_path: str) -> str:
    log_file_name = os.path.join(log_path, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = Path(log_file_name)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_name, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    return log_file_name


def database_building(config: dict | None = None) -> None:
    if config is None:
        config = load_database_config()

    log_file_name = _init_logging(config["log_path"])
    _prepare_directories(config)

    if os.path.exists(config["PlotCode_vb_path"]):
        logging.info("plot code vector database already exists: %s", config["PlotCode_vb_path"])
    else:
        plot_vdb = build_plotcode_vb(config)
        plot_data_count = len(plot_vdb.get()["ids"])
        logging.info("plot code vector database contains %s records", plot_data_count)

    if "cpu" not in config["database_device"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["database_device"].split(":")[-1]
        config["database_device"] = "cuda"

    downloaded_paper_title_list = []
    downloaded_paper_id_list = []
    downloaded_paper_key_list = []
    downloaded_paper_list = []
    downloaded_paper_information_bib = {}
    downloaded_paper_information = {}

    paper_crawler(
        config["bib_path"],
        config["save_pdf_path"],
        downloaded_paper_title_list,
        downloaded_paper_key_list,
        downloaded_paper_information,
        downloaded_paper_information_bib,
        config["api_parsing_method"],
        config["all_paper_num"],
    )

    logging.info("processing initial papers")
    used_experiment_pdf_path = []
    for root, _, files in os.walk(config["save_pdf_path"]):
        for file in files:
            file_path = os.path.join(root, file)
            used_experiment_pdf_path.append(file_path)

    pdf_extractor(
        used_experiment_pdf_path,
        config["save_md_path"],
        config["pdf_transform_mode"],
        config["api_parsing_method"],
        config.get("pdf_md_token"),
    )

    logging.info("building vector database for experiment papers")
    vectordb = build_paper_vb(
        used_experiment_pdf_path,
        downloaded_paper_information_bib,
        config["experiment_persist_directory_path"],
        config,
        log_file_name,
    )

    count_vector_database(vectordb)

    if os.path.exists(config["full_text_persist_directory_path"]):
        shutil.rmtree(config["full_text_persist_directory_path"])
    shutil.copytree(
        config["experiment_persist_directory_path"],
        config["full_text_persist_directory_path"],
    )

    field_list = [
        "references",
        "citations",
        "paperId",
        "title",
        "citationStyles",
        "externalIds",
        "openAccessPdf",
    ]

    for idx, paper_title in enumerate(downloaded_paper_title_list):
        sch = SemanticScholar(api_key=config["S2_API_KEY"])
        try:
            results = sch.search_paper(paper_title, limit=1, fields=field_list)
            time.sleep(1)

            if not (len(results) > 0):
                downloaded_paper_title_list[idx] = [paper_title]
                logging.error("paper not found in SemanticScholar: %s", paper_title)
                continue
            downloaded_paper_title_list[idx] = [results[0]["title"], paper_title]
            downloaded_paper_list.append(results[0])
            downloaded_paper_id_list.append(results[0]["paperId"])
        except Exception as exc:
            downloaded_paper_title_list[idx] = [paper_title]
            logging.error("SemanticScholar search failed for %s: %s", paper_title, exc)
            continue

    tmp_bib_path = os.path.join(os.path.dirname(config["vector_database_paper_information"]), "tmp_bib.bib")

    for current_paper in downloaded_paper_list:
        if len(downloaded_paper_title_list) >= config["all_paper_num"]:
            break

        tmp_bib = ""
        tmp_paper_id_list = []
        tmp_paper_title_list = []
        tmp_paper_key_list = []
        tmp_downloaded_paper_list = []

        logging.info("current paper references: %s", current_paper.title)
        if current_paper.references:
            for reference in current_paper.references[:50]:
                sch = SemanticScholar(api_key=config["S2_API_KEY"])

                try:
                    paper = sch.get_paper(reference["paperId"], fields=field_list)
                    time.sleep(1)
                except Exception as exc:
                    logging.error("get_paper failed for %s: %s", reference["title"], exc)
                    paper = None

                if not paper:
                    try:
                        results = sch.search_paper(reference["title"], limit=1, fields=field_list)
                        time.sleep(1)
                        if not (len(results) > 0):
                            logging.error("paper not found: %s", reference["title"])
                            continue
                        paper = results[0]
                    except Exception as exc:
                        logging.error("search_paper failed for %s: %s", reference["title"], exc)
                        continue

                parser = bibtexparser.bparser.BibTexParser()
                bib_db = bibtexparser.loads(paper.citationStyles["bibtex"], parser=parser)
                entry = bib_db.entries[0]

                if hasattr(paper, "externalIds") and paper["externalIds"]:
                    if hasattr(paper.externalIds, "DOI") and paper.externalIds["DOI"]:
                        entry["doi"] = paper.externalIds["DOI"]

                if hasattr(paper, "openAccessPdf") and paper["openAccessPdf"]:
                    if hasattr(paper.openAccessPdf, "url") and paper.openAccessPdf["url"]:
                        entry["url"] = paper.externalIds["url"]

                tmp_db = BibDatabase()
                tmp_db.entries = [entry]
                writer = bibtexparser.bwriter.BibTexWriter()
                bib = bibtexparser.dumps(tmp_db, writer=writer)

                if not check_reference_repeat(
                    [paper.title, reference["title"], entry["title"]],
                    paper["paperId"],
                    entry["ID"],
                    bib,
                    downloaded_paper_title_list,
                    downloaded_paper_id_list,
                    downloaded_paper_key_list,
                    downloaded_paper_information,
                    tmp_paper_title_list,
                    tmp_paper_id_list,
                    tmp_paper_key_list,
                    tmp_bib,
                    config,
                ):
                    tmp_bib += bib + "\n"
                    tmp_paper_id_list.append(paper["paperId"])
                    tmp_downloaded_paper_list.append(paper)
                    tmp_paper_title_list.append([paper.title, reference["title"], entry["title"]])
                    tmp_paper_key_list.append(entry["ID"])

        with open(tmp_bib_path, "w", encoding="utf-8") as f:
            f.write(tmp_bib)

        paper_crawler(
            tmp_bib_path,
            config["save_pdf_path"],
            downloaded_paper_title_list,
            downloaded_paper_key_list,
            downloaded_paper_information,
            downloaded_paper_information_bib,
            config["api_parsing_method"],
            config["all_paper_num"],
            tmp_paper_id_list,
            tmp_downloaded_paper_list,
            tmp_paper_title_list,
        )

        downloaded_paper_id_list.extend(tmp_paper_id_list)
        downloaded_paper_list.extend(tmp_downloaded_paper_list)

        if len(downloaded_paper_title_list) >= config["all_paper_num"]:
            break

        tmp_bib = ""
        tmp_paper_id_list = []
        tmp_paper_title_list = []
        tmp_paper_key_list = []
        tmp_downloaded_paper_list = []

        logging.info("current paper citations: %s", current_paper.title)
        if current_paper.citations:
            for citation in current_paper.citations[:50]:
                sch = SemanticScholar(api_key=config["S2_API_KEY"])

                try:
                    paper = sch.get_paper(citation["paperId"], fields=field_list)
                    time.sleep(1)
                except Exception as exc:
                    logging.error("get_paper failed for %s: %s", citation["title"], exc)
                    paper = None

                if not paper:
                    try:
                        results = sch.search_paper(citation["title"], limit=1, fields=field_list)
                        time.sleep(1)
                        if not (len(results) > 0):
                            logging.error("paper not found: %s", citation["title"])
                            continue
                        paper = results[0]
                    except Exception as exc:
                        logging.error("search_paper failed for %s: %s", citation["title"], exc)
                        continue

                parser = bibtexparser.bparser.BibTexParser()
                bib_db = bibtexparser.loads(paper.citationStyles["bibtex"], parser=parser)
                entry = bib_db.entries[0]

                if hasattr(paper, "externalIds") and paper["externalIds"]:
                    if hasattr(paper.externalIds, "DOI") and paper.externalIds["DOI"]:
                        entry["doi"] = paper.externalIds["DOI"]

                if hasattr(paper, "openAccessPdf") and paper["openAccessPdf"]:
                    if hasattr(paper.openAccessPdf, "url") and paper.openAccessPdf["url"]:
                        entry["url"] = paper.externalIds["url"]

                tmp_db = BibDatabase()
                tmp_db.entries = [entry]
                writer = bibtexparser.bwriter.BibTexWriter()
                bib = bibtexparser.dumps(tmp_db, writer=writer)

                if not check_reference_repeat(
                    [paper.title, citation["title"], entry["title"]],
                    paper["paperId"],
                    entry["ID"],
                    bib,
                    downloaded_paper_title_list,
                    downloaded_paper_id_list,
                    downloaded_paper_key_list,
                    downloaded_paper_information,
                    tmp_paper_title_list,
                    tmp_paper_id_list,
                    tmp_paper_key_list,
                    tmp_bib,
                    config,
                ):
                    tmp_bib += bib + "\n"
                    tmp_paper_id_list.append(paper["paperId"])
                    tmp_downloaded_paper_list.append(paper)
                    tmp_paper_title_list.append([paper.title, citation["title"], entry["title"]])
                    tmp_paper_key_list.append(entry["ID"])

        with open(tmp_bib_path, "w", encoding="utf-8") as f:
            f.write(tmp_bib)

        paper_crawler(
            tmp_bib_path,
            config["save_pdf_path"],
            downloaded_paper_title_list,
            downloaded_paper_key_list,
            downloaded_paper_information,
            downloaded_paper_information_bib,
            config["api_parsing_method"],
            config["all_paper_num"],
            tmp_paper_id_list,
            tmp_downloaded_paper_list,
            tmp_paper_title_list,
        )

    if os.path.exists(tmp_bib_path):
        os.remove(tmp_bib_path)

    with open(config["vector_database_paper_information"], "w", encoding="utf-8") as f:
        json.dump(downloaded_paper_information, f, indent=4, ensure_ascii=False)

    logging.info("processing full text papers")
    used_full_text_pdf_path = []
    for file in sorted(os.listdir(config["save_pdf_path"]))[len(used_experiment_pdf_path):]:
        used_full_text_pdf_path.append(os.path.join(config["save_pdf_path"], file))

    pdf_extractor(
        used_full_text_pdf_path,
        config["save_md_path"],
        config["pdf_transform_mode"],
        config["api_parsing_method"],
        config.get("pdf_md_token"),
    )

    vectordb = build_paper_vb(
        used_full_text_pdf_path,
        downloaded_paper_information_bib,
        config["full_text_persist_directory_path"],
        config,
        log_file_name,
    )

    count_vector_database(vectordb)
