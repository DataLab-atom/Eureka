from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from llm.env import get_env, load_env
from scientific_research_work.common.config import deep_merge, resolve_path


def load_base_config(
    config_path: str | None = None,
    problem_id: str | None = None,
    override: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Path, str]:
    load_env()
    root_path = Path(__file__).resolve().parents[2]
    config_dir = root_path / "config"
    config_file = Path(config_path) if config_path else config_dir / "config.yaml"

    with open(config_file, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    active_problem = problem_id or base_cfg.get("active_problem", "demo")
    problem_file = config_dir / "problems" / f"{active_problem}.yaml"
    if problem_file.exists():
        with open(problem_file, "r", encoding="utf-8") as f:
            problem_cfg = yaml.safe_load(f) or {}
    else:
        problem_cfg = {}

    cfg = deep_merge(base_cfg, problem_cfg)
    if override:
        cfg = deep_merge(cfg, override)

    paths_cfg = cfg.get("paths", {})
    root_override = paths_cfg.get("root")
    if root_override:
        root_path = Path(root_override)

    resolved_problem_id = cfg.get("problem_id", active_problem)
    return cfg, root_path, resolved_problem_id


def load_database_config(
    config_path: str | None = None,
    problem_id: str | None = None,
    override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg, root_path, problem_id = load_base_config(config_path=config_path, problem_id=problem_id, override=override)
    paths_cfg = cfg.get("paths", {})
    result_dir = paths_cfg.get("result_dir", "result")
    problems_dir = paths_cfg.get("problems_dir") or paths_cfg.get("test_data_dir", "problems")
    vector_db_dir = paths_cfg.get("vector_db_dir", "vector_database")
    plot_code_template_dir = paths_cfg.get("plot_code_template_dir", "plot_code_template")

    bib_path = cfg.get("bib_path", f"{problems_dir}/{problem_id}/reference.bib")

    db_cfg = cfg.get("database_building", {})
    llm_cfg = cfg.get("llm", {})

    return {
        "log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/database_building"),
        "bib_path": resolve_path(root_path, bib_path),
        "save_pdf_path": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/database_building/paper/pdf",
        ),
        "save_md_path": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/database_building/paper/vector_database_data",
        ),
        "vector_database_paper_information": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/database_building/vector_database_paper_information.json",
        ),
        "plot_code_template": resolve_path(root_path, plot_code_template_dir),
        "experiment_persist_directory_path": resolve_path(
            root_path,
            f"{vector_db_dir}/paper/{problem_id}/experiment",
        ),
        "full_text_persist_directory_path": resolve_path(
            root_path,
            f"{vector_db_dir}/paper/{problem_id}/full_text",
        ),
        "PlotCode_vb_path": resolve_path(root_path, f"{vector_db_dir}/plot"),
        "embedding_model": db_cfg.get("embedding_model", "BAAI/bge-m3"),
        "embedding_model_source": db_cfg.get("embedding_model_source", "api"),
        "pdf_transform_mode": db_cfg.get("pdf_transform_mode", "api"),
        "api_parsing_method": db_cfg.get("api_parsing_method", "file"),
        "text_data_processing": db_cfg.get(
            "text_data_processing",
            {"min_tokens": 1300, "max_tokens": 2400, "chunk_size": 700},
        ),
        "database_device": db_cfg.get("database_device", "cpu"),
        "chart_model_name": db_cfg.get("chart_model_name", "gpt-4.1-mini"),
        "judge_model_name": db_cfg.get("judge_model_name", "gpt-5-mini"),
        "temperature": db_cfg.get("temperature", llm_cfg.get("temperature", 1.0)),
        "all_paper_num": db_cfg.get("all_paper_num", 70),
        "S2_API_KEY": get_env("S2_API_KEY") or cfg.get("s2_api_key") or "",
        "pdf_md_token": cfg.get("pdf_md_token") or get_env("PDF_MD_TOKEN", ""),
    }


def load_experiment_config(
    config_path: str | None = None,
    problem_id: str | None = None,
    override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg, root_path, problem_id = load_base_config(config_path=config_path, problem_id=problem_id, override=override)
    paths_cfg = cfg.get("paths", {})
    result_dir = paths_cfg.get("result_dir", "result")
    problems_dir = paths_cfg.get("problems_dir") or paths_cfg.get("test_data_dir", "problems")
    vector_db_dir = paths_cfg.get("vector_db_dir", "vector_database")
    plot_code_template_dir = paths_cfg.get("plot_code_template_dir", "plot_code_template")

    experiment_cfg = cfg.get("experiment", {})
    drawing_cfg = cfg.get("drawing", {})
    writing_cfg = cfg.get("experiment_writing", {})
    llm_cfg = cfg.get("llm", {})
    db_cfg = cfg.get("database_building", {})

    embedding_model = experiment_cfg.get("embedding_model", db_cfg.get("embedding_model", "BAAI/bge-m3"))
    embedding_model_source = experiment_cfg.get("embedding_model_source", db_cfg.get("embedding_model_source", "api"))

    multimodal_model_name = (
        experiment_cfg.get("multimodal_model_name")
        or experiment_cfg.get("multimodel_model_name")
        or llm_cfg.get("multimodal_model_name", "gpt-4.1-mini")
    )

    return {
        "problem_id": problem_id,
        "log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/experimenting"),
        "experiment_log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/experimenting"),
        "drawing_log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/drawing"),
        "writing_log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/experiment_writing"),
        "bib_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/reference.bib"),
        "research_information": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/research_information.json",
        ),
        "experiment_code_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/experiment_code.py"),
        "method_code_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/code"),
        "reference_information": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/reference_information.json",
        ),
        "result_perspective_path": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/doing/perspective",
        ),
        "experimental_data_explain_path": resolve_path(
            root_path,
            f"{problems_dir}/{problem_id}/experimental_data_explain.md",
        ),
        "experimental_environment_path": resolve_path(
            root_path,
            f"{problems_dir}/{problem_id}/experimental_environment",
        ),
        "experiment_persist_directory_path": resolve_path(
            root_path,
            f"{vector_db_dir}/paper/{problem_id}/full_text",
        ),
        "save_md_path": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/database_building/paper/vector_database_data",
        ),
        "method_statement": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/method_statement.md",
        ),
        "reference_data": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/reference_data",
        ),
        "analysis_perspective_path": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/analysis_perspective_explain.json",
        ),
        "plot_code_template": resolve_path(root_path, plot_code_template_dir),
        "PlotCode_vb_path": resolve_path(root_path, f"{vector_db_dir}/plot"),
        "need_draw_perspectives": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/doing/perspective",
        ),
        "embedding_model": embedding_model,
        "embedding_model_source": embedding_model_source,
        "reranker_path": experiment_cfg.get("reranker_path", "BAAI/bge-reranker-v2-m3"),
        "reranker_model_source": experiment_cfg.get("reranker_model_source", "api"),
        "database_device": experiment_cfg.get("database_device", db_cfg.get("database_device", "cpu")),
        "text_model_name": experiment_cfg.get("text_model_name", llm_cfg.get("text_model_name", "gpt-4.1-mini")),
        "multimodal_model_name": multimodal_model_name,
        "multimodel_model_name": multimodal_model_name,
        "experiment_judge_model": experiment_cfg.get("judge_model_name", "gpt-4.1"),
        "temperature": experiment_cfg.get("temperature", llm_cfg.get("temperature", 1.0)),
        "retrieval_data_num": experiment_cfg.get("retrieval_data_num", 20),
        "reranked_data_num": experiment_cfg.get("reranked_data_num", 8),
        "all_analysis_perspective_num": experiment_cfg.get("analysis_perspective_num", 1),
        "retrieval_template_num": drawing_cfg.get("retrieval_template_num", 5),
        "max_try": drawing_cfg.get("max_try", 5),
        "mode_select": experiment_cfg.get("mode", "get"),
        "devices": experiment_cfg.get("devices", ["no_device", 2]),
        "perspectives": experiment_cfg.get("perspectives", ["analysis_perspective_000"]),
        "result_write_path": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experiment_writing/experiment.tex",
        ),
        "outline_result_path": resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experiment_writing/outline_result.md",
        ),
        "code_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/code"),
        "integration_model_name": writing_cfg.get("integration_model_name", "gpt-4.1-mini"),
        "tex_template_path": resolve_path(
            root_path,
            f"{problems_dir}/{problem_id}/template/template.tex",
        ),
        "writing_temperature": writing_cfg.get("temperature", 1.0),
    }


def load_writing_config(
    section: str,
    config_path: str | None = None,
    problem_id: str | None = None,
    override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg, root_path, problem_id = load_base_config(config_path, problem_id, override)
    paths_cfg = cfg.get("paths", {})
    result_dir = paths_cfg.get("result_dir", "result")
    problems_dir = paths_cfg.get("problems_dir") or paths_cfg.get("test_data_dir", "problems")
    vector_db_dir = paths_cfg.get("vector_db_dir", "vector_database")

    llm_cfg = cfg.get("llm", {})
    db_cfg = cfg.get("database_building", {})
    exp_cfg = cfg.get("experiment", {})

    method_cfg = cfg.get("method_writing", {})
    related_cfg = cfg.get("related_work_writing", {})
    intro_cfg = cfg.get("introduction_writing", {})

    if section == "method_writing":
        return {
            "problem_id": problem_id,
            "result_write_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/method_writing/method.tex",
            ),
            "research_information": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/research_information.json",
            ),
            "log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/method_writing"),
            "outline_result_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/method_writing/outline_result.md",
            ),
            "our_method_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/code/Ours.py"),
            "tex_template_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/template/template.tex"),
            "improve_round": method_cfg.get("improve_round", 2),
            "write_model_name": method_cfg.get("write_model_name", llm_cfg.get("text_model_name", "gpt-4.1-mini")),
            "integration_model_name": method_cfg.get(
                "integration_model_name", llm_cfg.get("text_model_name", "gpt-4.1-mini")
            ),
            "temperature": method_cfg.get("temperature", llm_cfg.get("temperature", 1.0)),
        }

    embedding_model = related_cfg.get(
        "embedding_model",
        exp_cfg.get("embedding_model", db_cfg.get("embedding_model", "BAAI/bge-m3")),
    )
    embedding_model_source = related_cfg.get(
        "embedding_model_source",
        exp_cfg.get("embedding_model_source", db_cfg.get("embedding_model_source", "api")),
    )
    reranker_path = related_cfg.get("reranker_path", exp_cfg.get("reranker_path", "BAAI/bge-reranker-v2-m3"))
    reranker_model_source = related_cfg.get("reranker_model_source", exp_cfg.get("reranker_model_source", "api"))
    database_device = related_cfg.get(
        "database_device",
        exp_cfg.get("database_device", db_cfg.get("database_device", "cpu")),
    )

    if section == "related_work_writing":
        return {
            "problem_id": problem_id,
            "result_write_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/related_work_writing/related_work.tex",
            ),
            "research_information": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/research_information.json",
            ),
            "our_method_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/code/Ours.py"),
            "bib_result_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/related_work_writing/reference_bib.bib",
            ),
            "bib_result_list_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/related_work_writing/related_work_reference_bib.json",
            ),
            "log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/related_work_writing"),
            "tex_template_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/template/template.tex"),
            "persist_directory_path": resolve_path(root_path, f"{vector_db_dir}/paper/{problem_id}/full_text"),
            "text_model_name": related_cfg.get("text_model_name", llm_cfg.get("text_model_name", "gpt-4.1-mini")),
            "multimodal_model_name": related_cfg.get(
                "multimodal_model_name", llm_cfg.get("multimodal_model_name", "gpt-4.1-mini")
            ),
            "judge_model_name": related_cfg.get("judge_model_name", "gpt-4.1-mini"),
            "embedding_model": embedding_model,
            "embedding_model_source": embedding_model_source,
            "reranker_path": reranker_path,
            "reranker_model_source": reranker_model_source,
            "database_device": database_device,
            "retrieval_data_num": related_cfg.get("retrieval_data_num", 5),
            "field_num": related_cfg.get("field_num", 3),
            "sentence_num": related_cfg.get("sentence_num", 2),
            "used_cite_num": related_cfg.get("used_cite_num", 3),
            "used_data_num": related_cfg.get("used_data_num", 3),
            "temperature": related_cfg.get("temperature", llm_cfg.get("temperature", 1.0)),
        }

    if section == "introduction_writing":
        embedding_model = intro_cfg.get(
            "embedding_model",
            exp_cfg.get("embedding_model", db_cfg.get("embedding_model", "BAAI/bge-m3")),
        )
        embedding_model_source = intro_cfg.get(
            "embedding_model_source",
            exp_cfg.get("embedding_model_source", db_cfg.get("embedding_model_source", "api")),
        )
        reranker_path = intro_cfg.get("reranker_path", exp_cfg.get("reranker_path", "BAAI/bge-reranker-v2-m3"))
        reranker_model_source = intro_cfg.get("reranker_model_source", exp_cfg.get("reranker_model_source", "api"))
        database_device = intro_cfg.get(
            "database_device",
            exp_cfg.get("database_device", db_cfg.get("database_device", "cpu")),
        )

        return {
            "problem_id": problem_id,
            "result_write_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/introduction_writing/introduction.tex",
            ),
            "our_method_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/code/Ours.py"),
            "research_information": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/experimenting/get/research_information.json",
            ),
            "bib_result_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/introduction_writing/reference_bib.bib",
            ),
            "used_reference_bib_list_path": resolve_path(
                root_path,
                f"{result_dir}/{problem_id}/pipeline_process_data/related_work_writing/related_work_reference_bib.json",
            ),
            "log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/introduction_writing"),
            "tex_template_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/template/template.tex"),
            "persist_directory_path": resolve_path(root_path, f"{vector_db_dir}/paper/{problem_id}/full_text"),
            "text_model_name": intro_cfg.get("text_model_name", llm_cfg.get("text_model_name", "gpt-4.1-mini")),
            "multimodal_model_name": intro_cfg.get(
                "multimodal_model_name", llm_cfg.get("multimodal_model_name", "gpt-4.1-mini")
            ),
            "integration_model_name": intro_cfg.get(
                "integration_model_name", llm_cfg.get("text_model_name", "gpt-4.1-mini")
            ),
            "embedding_model": embedding_model,
            "embedding_model_source": embedding_model_source,
            "reranker_path": reranker_path,
            "reranker_model_source": reranker_model_source,
            "database_device": database_device,
            "retrieval_data_num": intro_cfg.get("retrieval_data_num", 8),
            "used_data_num": intro_cfg.get("used_data_num", 4),
            "temperature": intro_cfg.get("temperature", llm_cfg.get("temperature", 1.0)),
        }

    if section == "merge_writing":
        merge_cfg = cfg.get("merge_writing", {})

        introduction_tex = resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/introduction_writing/introduction.tex",
        )
        related_work_tex = resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/related_work_writing/related_work.tex",
        )
        method_tex = resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/method_writing/method.tex",
        )
        experiment_tex = resolve_path(
            root_path,
            f"{result_dir}/{problem_id}/pipeline_process_data/experiment_writing/experiment.tex",
        )

        return {
            "problem_id": problem_id,
            "chapter_paths": [introduction_tex, related_work_tex, method_tex, experiment_tex],
            "introduction_tex_path": introduction_tex,
            "related_work_tex_path": related_work_tex,
            "method_tex_path": method_tex,
            "experiment_tex_path": experiment_tex,
            "bib_source_paths": [
                resolve_path(
                    root_path,
                    f"{result_dir}/{problem_id}/pipeline_process_data/related_work_writing/reference_bib.bib",
                ),
                resolve_path(
                    root_path,
                    f"{result_dir}/{problem_id}/pipeline_process_data/introduction_writing/reference_bib.bib",
                ),
            ],
            "bib_result_path": resolve_path(root_path, f"{result_dir}/{problem_id}/reference_bib.bib"),
            "final_result_path": resolve_path(root_path, f"{result_dir}/{problem_id}/paper.tex"),
            "log_path": resolve_path(root_path, f"{result_dir}/{problem_id}/log/merge_writing"),
            "tex_template_path": resolve_path(root_path, f"{problems_dir}/{problem_id}/template/template.tex"),
            "style_guide": resolve_path(root_path, f"{problems_dir}/{problem_id}/style_guide.md"),
            "write_model_name": merge_cfg.get("write_model_name", llm_cfg.get("text_model_name", "gpt-4.1-mini")),
            "judge_model": merge_cfg.get("judge_model", merge_cfg.get("judge_model_name", "gpt-4.1")),
            "temperature": merge_cfg.get("temperature", llm_cfg.get("temperature", 1.0)),
            "improve_round": merge_cfg.get("improve_round", 2),
            "check_round": merge_cfg.get("check_round", 0),
            "evol_iter": merge_cfg.get("evol_iter", 2),
            "population": merge_cfg.get("population", 3),
        }

    raise ValueError(f"Unknown writing section: {section}")
