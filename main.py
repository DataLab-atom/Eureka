import argparse

from scientific_research_work.pipeline import PIPELINE_STEPS, run_pipeline


def _parse_steps(raw_steps: list[str] | None) -> list[str] | None:
    if not raw_steps:
        return None
    if len(raw_steps) == 1 and "," in raw_steps[0]:
        return [step.strip() for step in raw_steps[0].split(",") if step.strip()]
    return raw_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scientific_research_work pipeline")
    parser.add_argument(
        "--steps",
        nargs="*",
        help=f"Pipeline steps to run. Available: {', '.join(PIPELINE_STEPS)}",
    )
    parser.add_argument("--config", dest="config_path", help="Path to config.yaml")
    parser.add_argument("--problem", dest="problem_id", help="Problem id override")
    parser.add_argument("--list-steps", action="store_true", help="List available pipeline steps and exit")

    args = parser.parse_args()
    if args.list_steps:
        print("\n".join(PIPELINE_STEPS))
        return

    steps = _parse_steps(args.steps)
    run_pipeline(config_path=args.config_path, problem_id=args.problem_id, steps=steps)


if __name__ == "__main__":
    main()
