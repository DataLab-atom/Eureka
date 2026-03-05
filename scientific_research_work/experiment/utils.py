import logging
import os
import subprocess


def run_tmp_script(code_content: str, code_path: str, analysis_angle: str | None = None):
    if analysis_angle:
        logging.info("analysis angle %s code:\n%s", analysis_angle, code_content)

    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code_content)

    result = subprocess.run(
        ["python", code_path],
        capture_output=True,
        text=True,
    )
    return result
