import json
import logging
import os
import shutil
from functools import partial
from multiprocessing import Manager, Pool

from scientific_research_work.common.judge_evol.judge.agent_as_a_judge.text_eval_agent import TextJudgeAgent

from llm.client import LLMClient
from llm.message import system, text_part, user_content
from scientific_research_work.common.logging import setup_logging
from scientific_research_work.experiment.utils import run_tmp_script
from scientific_research_work.experiment.pipeline.analysis_perspective import get_method_exlain
from scientific_research_work.experiment.prompts.experimenting import (
    EXPERIMENT_ANALYSIS,
    EXPERIMENT_ANALYSIS_PROMPT,
    EXPERIMENT_ANALYSIS_SYSTEM,
)


def _build_messages(experimental_requirement: str, method_explain: str, refer_experiment_code: str, experimental_data_explain: str):
    prompt_text = EXPERIMENT_ANALYSIS_PROMPT.format(
        ANALYTICAL_PERSPECTIVE=experimental_requirement,
        EXPERIMENT_CODE=refer_experiment_code,
        EXPERIMENT_DATA_EXPLAIN=experimental_data_explain,
        METHOD_CODE=method_explain,
    )
    return [
        system(EXPERIMENT_ANALYSIS_SYSTEM),
        user_content([text_part(prompt_text)]),
    ]


def _generate_experiment_code(
    model_name: str,
    temperature: float,
    messages: list,
) -> str:
    client = LLMClient()
    response = client.call(
        messages=messages,
        model=model_name,
        temperature=temperature,
        response_schema=EXPERIMENT_ANALYSIS,
    )
    return response["experiment_analysis"]


def doing_experiment(
    analysis_perspective: dict,
    draw_data_save_path: str,
    config: dict,
    method_explain: str,
    experimental_data_explain: str,
    refer_experiment_code: str,
    device_queue,
):
    if device_queue:
        device_id = device_queue.get()
    else:
        device_id = None

    experimental_requirement = f"""
# Analysis perspective: {analysis_perspective['analysis_perspective']}
    - Core focus: {analysis_perspective['core_focus']}
    - Differentiated feature: {analysis_perspective['differentiated_feature']}
    - Expected insight: {analysis_perspective['expected_insight']}
    - Experimental metric:{analysis_perspective['experimental_metric']}
    """.strip()

    analysis_perspective_num = os.path.basename(os.path.dirname(draw_data_save_path))

    new_experimental_environment_path = os.path.join(os.path.dirname(draw_data_save_path), "experimental_environment")
    old_experimental_environment_path = config["experimental_environment_path"]

    experiment_code_path = os.path.join(new_experimental_environment_path, "experimental_code.py")
    experimental_result_data_path = os.path.join(new_experimental_environment_path, "experimental_result_data")

    if device_id:
        experimental_environment_code = (
            "import os\nimport sys\n"
            "sys.path.append(os.path.dirname(__file__))\n"
            "os.chdir(os.path.dirname(__file__))\n"
            f"os.environ['CUDA_VISIBLE_DEVICES']='{device_id}'\n"
        )
    else:
        experimental_environment_code = (
            "import os\nimport sys\n"
            "sys.path.append(os.path.dirname(__file__))\n"
            "os.chdir(os.path.dirname(__file__))\n"
        )

    messages = _build_messages(experimental_requirement, method_explain, refer_experiment_code, experimental_data_explain)

    current_try_num = 0
    while current_try_num < 3:
        try:
            experiment_code = _generate_experiment_code(
                config["text_model_name"],
                config["temperature"],
                messages,
            )
        except Exception as exc:
            current_try_num += 1
            logging.error("experiment generation failed: %s", exc)
            continue

        experimental_code = experimental_environment_code + experiment_code

        if os.path.exists(new_experimental_environment_path):
            shutil.rmtree(new_experimental_environment_path)
        shutil.copytree(old_experimental_environment_path, new_experimental_environment_path)
        os.makedirs(experimental_result_data_path, exist_ok=True)

        result = run_tmp_script(
            experimental_code,
            experiment_code_path,
            os.path.split(new_experimental_environment_path)[0].split("/")[-1],
        )

        if result.returncode != 0:
            error_msg = f"关于分析角度:**{analysis_perspective_num}**的实验代码运行报错:\n{result.stderr}."
            logging.error(error_msg)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        text_part(
                            f"Previous code attempt failed. Error: {result.stderr}\n"
                            "Please generate corrected code to fulfill the demand"
                        )
                    ],
                }
            )
            current_try_num += 1
            continue

        if os.path.exists(os.path.join(config["result_perspective_path"], analysis_perspective_num, "experimental_code.py")):
            os.remove(os.path.join(config["result_perspective_path"], analysis_perspective_num, "experimental_code.py"))

        if os.path.exists(draw_data_save_path):
            shutil.rmtree(draw_data_save_path)

        os.rename(
            experiment_code_path,
            os.path.join(config["result_perspective_path"], analysis_perspective_num, "experimental_code.py"),
        )
        os.rename(experimental_result_data_path, draw_data_save_path)

        logging.info("judging experimental code")

        questions = [
            f"May I ask if the experimental code provided is designed according to the following analytical perspective: \n{experimental_requirement}",
            "May I ask if the Python code implementation of the following methods is correctly reproduced in the provided experimental code? (Only focus on reproducing the core methods at the logical level(keeping the core mechanism unchanged), ignoring any code refactoring that does not affect the output results of the core algorithm, such as logging and naming styles. **And note: if the inability to fully reproduce these methods is due to the requirements of the experimental analysis perspective, it can be disregarded**) \n"
            + method_explain,
            "May I ask if the provided experimental code used the provided experimental data or its data generation method instead of using another data or generation method?\n**And note: If different experimental data generation methods are used due to the requirement of experimental analysis perspective, they can be ignored**\n The provided experimental data or its data generation method is as follows:\n"
            + experimental_data_explain,
        ]

        with open(
            os.path.join(config["result_perspective_path"], analysis_perspective_num, "experimental_code.py"),
            "r",
            encoding="utf-8",
        ) as f:
            experiment_code = f.read()

        infoBench_eval_agent = TextJudgeAgent(config["experiment_judge_model"])
        judge_result = infoBench_eval_agent.evaluate(
            f"The experiment code:\n```python\n{experiment_code}\n```",
            questions,
        )

        if all(judge_result["scores"]):
            logging.info("judge passed: %s", analysis_perspective_num)
            break

        current_try_num += 1
        logging.error("judge failed: %s (try %s)", analysis_perspective_num, current_try_num)
        logging.info("reasons: %s", judge_result["reasons"])
        logging.info("suggestions: %s", judge_result["suggestions"])

    if device_queue:
        device_queue.put(device_id)


def run_experiments(config: dict, perspectives: list, devices: list, log_file_name: str | None = None):
    method_explain, _method_statement = get_method_exlain(config)

    analysis_path = config.get("analysis_perspective_path") or os.path.join(
        os.path.dirname(config["reference_data"]), "analysis_perspective_explain.json"
    )
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_perspective_set = json.load(f)

    select_perspective_list = []
    draw_data_save_path_list = []

    for perspective in perspectives:
        analysis_perspective = analysis_perspective_set[perspective]
        draw_data_save_path_list.append(
            os.path.join(config["result_perspective_path"], perspective, "experimental_result_data")
        )
        select_perspective_list.append(analysis_perspective)

    with open(config["experimental_data_explain_path"], "r", encoding="utf-8") as f:
        experimental_data_explain = f.read()

    with open(config["experiment_code_path"], "r", encoding="utf-8") as f:
        refer_experiment_code = f.read()

    if devices[0] != "no_device":
        with Manager() as manager:
            device_queue = manager.Queue()
            for dev in devices:
                device_queue.put(dev)

            partial_doing_experiment = partial(
                doing_experiment,
                config=config,
                method_explain=method_explain,
                experimental_data_explain=experimental_data_explain,
                refer_experiment_code=refer_experiment_code,
                device_queue=device_queue,
            )
            if log_file_name:
                pool_kwargs = {"initializer": setup_logging, "initargs": (log_file_name,)}
            else:
                pool_kwargs = {}
            with Pool(processes=len(devices), **pool_kwargs) as pool:
                pool.starmap(partial_doing_experiment, zip(select_perspective_list, draw_data_save_path_list))
    else:
        partial_doing_experiment = partial(
            doing_experiment,
            config=config,
            method_explain=method_explain,
            experimental_data_explain=experimental_data_explain,
            refer_experiment_code=refer_experiment_code,
            device_queue=None,
        )
        if log_file_name:
            pool_kwargs = {"initializer": setup_logging, "initargs": (log_file_name,)}
        else:
            pool_kwargs = {}
        with Pool(processes=int(devices[1]), **pool_kwargs) as pool:
            pool.starmap(partial_doing_experiment, zip(select_perspective_list, draw_data_save_path_list))

    logging.info("experiment generation finished")
