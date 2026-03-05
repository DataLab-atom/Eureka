import datetime
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scientific_research_work.common.logging import setup_logging
from scientific_research_work.experiment.pipeline.analysis_perspective import generate_analysis_perspectives
from scientific_research_work.experiment.pipeline.drawing import drawing
from scientific_research_work.experiment.pipeline.experiment_generation import run_experiments
from scientific_research_work.experiment.pipeline.writing import write_experiment
from scientific_research_work.common.settings import load_experiment_config


def _init_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_path = Path(log_file_name)
    log_path.touch(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_name, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    return log_file_name


def run_analysis(config: dict | None = None) -> str:
    cfg = config or load_experiment_config()
    log_file_name = _init_logging(cfg["experiment_log_path"])
    generate_analysis_perspectives(cfg)
    return log_file_name


def run_doing(
    config: dict | None = None,
    perspectives: list | None = None,
    devices: list | None = None,
) -> str:
    cfg = config or load_experiment_config()
    log_file_name = _init_logging(cfg["experiment_log_path"])
    run_experiments(
        cfg,
        perspectives or cfg.get("perspectives", ["analysis_perspective_000"]),
        devices or cfg.get("devices", ["no_device", 2]),
        log_file_name,
    )
    return log_file_name


def run_draw(config: dict | None = None, perspectives: list | None = None) -> str:
    cfg = config or load_experiment_config()
    log_file_name = _init_logging(cfg["drawing_log_path"])
    drawing(cfg, perspectives or cfg.get("perspectives", ["analysis_perspective_000"]), log_file_name)
    return log_file_name


def run_write(config: dict | None = None) -> None:
    cfg = config or load_experiment_config()
    _init_logging(cfg["writing_log_path"])
    write_experiment(cfg)


def run_all(
    config: dict | None = None,
    perspectives: list | None = None,
    devices: list | None = None,
) -> str:
    cfg = config or load_experiment_config()
    log_file_name = _init_logging(cfg["experiment_log_path"])
    generate_analysis_perspectives(cfg)
    run_experiments(
        cfg,
        perspectives or cfg.get("perspectives", ["analysis_perspective_000"]),
        devices or cfg.get("devices", ["no_device", 2]),
        log_file_name,
    )
    drawing(cfg, perspectives or cfg.get("perspectives", ["analysis_perspective_000"]), log_file_name)
    _init_logging(cfg["writing_log_path"])
    write_experiment(cfg)
    return log_file_name


def experiment(
    config: dict | None = None,
    mode: str | None = None,
    perspectives: list | None = None,
    devices: list | None = None,
):
    cfg = config or load_experiment_config()
    resolved_mode = mode or cfg.get("mode_select", "get")
    if resolved_mode == "get":
        return run_analysis(cfg)
    if resolved_mode == "doing":
        return run_doing(cfg, perspectives, devices)
    if resolved_mode == "draw":
        return run_draw(cfg, perspectives)
    if resolved_mode == "write":
        return run_write(cfg)
    if resolved_mode == "all":
        return run_all(cfg, perspectives, devices)
    raise ValueError(f"unknown mode: {resolved_mode}")
