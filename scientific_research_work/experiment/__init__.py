def experiment(*args, **kwargs):
    from scientific_research_work.experiment.pipeline.orchestrator import experiment as _experiment

    return _experiment(*args, **kwargs)


def run_analysis(*args, **kwargs):
    from scientific_research_work.experiment.pipeline.orchestrator import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)


def run_doing(*args, **kwargs):
    from scientific_research_work.experiment.pipeline.orchestrator import run_doing as _run_doing

    return _run_doing(*args, **kwargs)


def run_draw(*args, **kwargs):
    from scientific_research_work.experiment.pipeline.orchestrator import run_draw as _run_draw

    return _run_draw(*args, **kwargs)


def run_write(*args, **kwargs):
    from scientific_research_work.experiment.pipeline.orchestrator import run_write as _run_write

    return _run_write(*args, **kwargs)


def run_all(*args, **kwargs):
    from scientific_research_work.experiment.pipeline.orchestrator import run_all as _run_all

    return _run_all(*args, **kwargs)


__all__ = [
    "experiment",
    "run_analysis",
    "run_doing",
    "run_draw",
    "run_write",
    "run_all",
]
