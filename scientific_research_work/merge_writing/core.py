from __future__ import annotations

import datetime
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter

from scientific_research_work.common.judge_evol.run_latex_evolution import evolve_paper
from llm.client import LLMClient
from llm.message import system, text_part, user_content
from scientific_research_work.common.fs import ensure_dir
from scientific_research_work.common.logging import setup_logging
from scientific_research_work.common.settings import load_writing_config

from .prompts import (
    ABSTRACT_WRITE_SYSTEM,
    CHECK_CITE_SYSTEM,
    CHECK_SUBMIT_SYSTEM,
    CONCLUSION_WRITE_SYSTEM,
    INIT_MERGE_PROMPT,
    INIT_MERGE_SYSTEM,
    MERGE_IMPROVE_SYSTEM,
    TITLE_WRITE_SYSTEM,
)


CHAPTER_KEYS = (
    "introduction_tex_path",
    "related_work_tex_path",
    "method_tex_path",
    "experiment_tex_path",
)


def doing_makedirs(config: dict) -> None:
    ensure_dir(config["log_path"])
    ensure_dir(os.path.dirname(config["final_result_path"]))
    ensure_dir(os.path.dirname(config["bib_result_path"]))


def _init_logging(log_path: str) -> str:
    ensure_dir(log_path)
    log_file_name = os.path.join(log_path, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = Path(log_file_name)
    log_file.touch(exist_ok=True)
    setup_logging(log_file_name)
    return log_file_name


def _extract_field(response: object, key: str) -> str:
    if isinstance(response, dict) and key in response:
        return response[key]
    if isinstance(response, str):
        return response
    return str(response)


def _call_json(model_name: str, temperature: float, system_prompt: str, parts: list[dict]) -> dict:
    client = LLMClient()
    messages = [system(system_prompt), user_content(parts)]
    return client.call(messages, model=model_name, temperature=temperature, json_mode=True)


def _call_tex_code(model_name: str, temperature: float, system_prompt: str, content: str) -> str:
    response = _call_json(model_name, temperature, system_prompt, [text_part(content)])
    return _extract_field(response, "tex_code")


def _call_result(model_name: str, temperature: float, system_prompt: str, content: str) -> str:
    response = _call_json(model_name, temperature, system_prompt, [text_part(content)])
    return _extract_field(response, "result")


def _collect_chapters(config: dict) -> str:
    chapter_paths = config.get("chapter_paths")
    if not chapter_paths:
        chapter_paths = [config[key] for key in CHAPTER_KEYS]

    done_write = ""
    for chapter in chapter_paths:
        if not os.path.exists(chapter):
            raise FileNotFoundError(f"chapter not found: {chapter}")
        with open(chapter, "r", encoding="utf-8") as f:
            done_write += f.read() + "\n\n"

    return done_write


def _merge_bib_sources(bib_paths: Iterable[str]) -> str:
    entries = {}
    raw_parts: List[str] = []

    parser = bibtexparser.bparser.BibTexParser()
    writer = BibTexWriter()

    for path in bib_paths:
        if not path or not os.path.exists(path):
            logging.warning("bib source missing: %s", path)
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue
        try:
            bib_db = bibtexparser.loads(content, parser=parser)
            for entry in bib_db.entries:
                entry_id = entry.get("ID")
                if entry_id and entry_id not in entries:
                    entries[entry_id] = entry
        except Exception as exc:
            logging.warning("failed to parse bib %s: %s", path, exc)
            raw_parts.append(content)

    merged = ""
    if entries:
        db = BibDatabase()
        db.entries = list(entries.values())
        merged = bibtexparser.dumps(db, writer=writer).strip()

    if raw_parts:
        merged = (merged + "\n\n" if merged else "") + "\n\n".join(raw_parts)

    return merged.strip()


def _load_bib_content(config: dict) -> str:
    bib_sources = config.get("bib_source_paths") or []
    if bib_sources:
        bib_content = _merge_bib_sources(bib_sources)
        if config.get("bib_result_path"):
            ensure_dir(os.path.dirname(config["bib_result_path"]))
            with open(config["bib_result_path"], "w", encoding="utf-8") as f:
                f.write(bib_content)
        return bib_content

    bib_path = config.get("bib_result_path")
    if bib_path and os.path.exists(bib_path):
        with open(bib_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def abstract_conclusion_title_writing(config: dict, done_write: str) -> tuple[str, str]:
    conclusion = _call_tex_code(
        config["write_model_name"],
        config["temperature"],
        CONCLUSION_WRITE_SYSTEM,
        done_write,
    )
    done_write += "\n\n" + conclusion
    logging.info("Conclusion content:\n%s", conclusion)

    abstract = _call_tex_code(
        config["write_model_name"],
        config["temperature"],
        ABSTRACT_WRITE_SYSTEM,
        done_write,
    )
    logging.info("Abstract content:\n%s", abstract)

    done_write = abstract + "\n\n" + done_write

    title = _call_tex_code(
        config["write_model_name"],
        config["temperature"],
        TITLE_WRITE_SYSTEM,
        done_write,
    )
    logging.info("Title content:\n%s", title)

    return done_write, title


def extract_specific_parts(latex_text: str) -> tuple[str, str]:
    intro_pattern = r"\\section\s*\{\s*Introduction\s*\}"
    parts = re.split(intro_pattern, latex_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) > 1:
        before_intro = parts[0].strip()
    else:
        raise RuntimeError("Failed to extract content before Introduction section.")

    concl_pattern = r"\\section\s*\{\s*Conclusion\s*\}"
    match = re.search(concl_pattern, latex_text, flags=re.IGNORECASE)
    if match:
        from_conclusion = latex_text[match.start():].strip()
    else:
        raise RuntimeError("Failed to extract content from Conclusion section.")

    return before_intro, from_conclusion


def other_check(config: dict, improve_merge_content: str, bib_content: str) -> tuple[str, str]:
    for idx in range(config["check_round"]):
        checked_submit_content = _call_tex_code(
            config["write_model_name"],
            config["temperature"],
            CHECK_SUBMIT_SYSTEM,
            improve_merge_content,
        )
        if checked_submit_content.strip() == "already completed":
            break
        improve_merge_content = checked_submit_content
        logging.info("Check round %s result:\n%s", idx + 1, improve_merge_content)

    style_guide = None
    if config.get("style_guide") and os.path.exists(config["style_guide"]):
        with open(config["style_guide"], "r", encoding="utf-8") as f:
            style_guide = f.read()

    before_intro, from_conclusion = extract_specific_parts(improve_merge_content)
    paper_main_content = evolve_paper(improve_merge_content, config, style_guide)
    improve_merge_content = f"{before_intro}\n\n{paper_main_content}\n\n{from_conclusion}"

    bib_content = _call_result(
        config["write_model_name"],
        config["temperature"],
        CHECK_CITE_SYSTEM,
        f"# Paper content:\n```\n{improve_merge_content}\n```\n# References cited(`reference_bib.bib` file content):\n{bib_content}\n",
    )

    return improve_merge_content, bib_content


def merge_writing(
    config: Optional[dict] = None,
    done_write: Optional[str] = None,
    bib_content: Optional[str] = None,
) -> tuple[str, str]:
    cfg = config or load_writing_config("merge_writing")
    _init_logging(cfg["log_path"])

    if done_write is None:
        done_write = _collect_chapters(cfg)

    done_write, paper_title = abstract_conclusion_title_writing(cfg, done_write)

    with open(cfg["tex_template_path"], "r", encoding="utf-8") as f:
        tex_template = f.read()

    if bib_content is None:
        bib_content = _load_bib_content(cfg)

    logging.info("Content before merge:\n%s", done_write)

    init_merge_content = _call_tex_code(
        cfg["write_model_name"],
        cfg["temperature"],
        INIT_MERGE_SYSTEM.format(TEX_TEMPLATE=tex_template),
        INIT_MERGE_PROMPT.format(TITLE=paper_title, CONTENT=done_write),
    )
    logging.info("Initial merged content:\n%s", init_merge_content)

    improve_merge_content = init_merge_content
    for idx in range(cfg["improve_round"]):
        improved = _call_tex_code(
            cfg["write_model_name"],
            cfg["temperature"],
            MERGE_IMPROVE_SYSTEM.format(STY_FORMAT_DEMAND=tex_template),
            init_merge_content,
        )

        if ("already completed" in improved) or (cfg["improve_round"] - 1 == idx):
            improve_merge_content = init_merge_content
            break
        init_merge_content = improved
        logging.info("Improve round %s result:\n%s", idx + 1, init_merge_content)

    logging.info("Final improved content:\n%s", improve_merge_content)

    final_tex, bib_content = other_check(cfg, improve_merge_content, bib_content)

    ensure_dir(os.path.dirname(cfg["final_result_path"]))
    with open(cfg["final_result_path"], "w", encoding="utf-8") as f:
        f.write(final_tex)

    ensure_dir(os.path.dirname(cfg["bib_result_path"]))
    with open(cfg["bib_result_path"], "w", encoding="utf-8") as f:
        f.write(bib_content)

    logging.info("Merge completed, result saved to %s", cfg["final_result_path"])

    return final_tex, bib_content
