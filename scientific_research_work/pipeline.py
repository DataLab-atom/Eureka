from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from scientific_research_work.common.logging import setup_logging
from scientific_research_work.common.settings import (
    load_database_config,
    load_experiment_config,
    load_writing_config,
)
from scientific_research_work.database_building.core import database_building
from scientific_research_work.experiment.pipeline.orchestrator import run_all as run_experiment_all
from scientific_research_work.introduction_writing.core import doing_makedirs as intro_makedirs
from scientific_research_work.introduction_writing.core import write_introduction
from scientific_research_work.merge_writing.core import doing_makedirs as merge_makedirs
from scientific_research_work.merge_writing.core import merge_writing
from scientific_research_work.method_writing.core import doing_makedirs as method_makedirs
from scientific_research_work.method_writing.core import write_methodology
from scientific_research_work.related_work_writing.core import doing_makedirs as related_makedirs
from scientific_research_work.related_work_writing.core import write_related_work

PIPELINE_STEPS = (
    "database_building",
    "experiment",
    "method_writing",
    "related_work_writing",
    "introduction_writing",
    "merge_writing",
)


def _init_logging(config_path: str | None, problem_id: str | None) -> str:
    db_cfg = load_database_config(config_path=config_path, problem_id=problem_id)
    base_log_dir = Path(db_cfg["log_path"]).parent
    log_dir = base_log_dir / "pipeline"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(str(log_file))
    return str(log_file)


def _normalize_steps(steps: Optional[Iterable[str]]) -> List[str]:
    if steps is None:
        return list(PIPELINE_STEPS)
    if isinstance(steps, str):
        steps = [s.strip() for s in steps.split(",") if s.strip()]
    normalized = []
    for step in steps:
        if step not in PIPELINE_STEPS:
            raise ValueError(f"unknown step: {step}")
        if step not in normalized:
            normalized.append(step)
    return normalized


def run_pipeline(
    *,
    config_path: str | None = None,
    problem_id: str | None = None,
    steps: Iterable[str] | None = None,
) -> None:
    log_file = _init_logging(config_path, problem_id)
    logging.info("pipeline start, log=%s", log_file)

    selected = _normalize_steps(steps)
    logging.info("pipeline steps: %s", selected)

    if "database_building" in selected:
        logging.info("pipeline: database_building")
        db_cfg = load_database_config(config_path=config_path, problem_id=problem_id)
        database_building(db_cfg)

    if "experiment" in selected:
        logging.info("pipeline: experiment")
        exp_cfg = load_experiment_config(config_path=config_path, problem_id=problem_id)
        run_experiment_all(exp_cfg)

    if "method_writing" in selected:
        logging.info("pipeline: method_writing")
        method_cfg = load_writing_config("method_writing", config_path, problem_id)
        method_makedirs(method_cfg)
        write_methodology(method_cfg)

    if "related_work_writing" in selected:
        logging.info("pipeline: related_work_writing")
        related_cfg = load_writing_config("related_work_writing", config_path, problem_id)
        related_makedirs(related_cfg)
        write_related_work(related_cfg)

    if "introduction_writing" in selected:
        logging.info("pipeline: introduction_writing")
        intro_cfg = load_writing_config("introduction_writing", config_path, problem_id)
        intro_makedirs(intro_cfg)
        write_introduction(intro_cfg)

    if "merge_writing" in selected:
        logging.info("pipeline: merge_writing")
        merge_cfg = load_writing_config("merge_writing", config_path, problem_id)
        merge_makedirs(merge_cfg)
        merge_writing(merge_cfg)

    logging.info("pipeline done")
