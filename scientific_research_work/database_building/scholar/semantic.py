import logging

from scientific_research_work.common.judge_evol.judge.agent_as_a_judge.text_eval_agent import TextJudgeAgent


def check_reference_repeat(
    titles: list,
    paper_id: str,
    key_id: str,
    bib: str,
    downloaded_paper_title_list: list,
    downloaded_paper_id_list: list,
    downloaded_paper_key_list: list,
    downloaded_paper_information: dict,
    batch_paper_title_list: list[list],
    batch_paper_paper_id_list: list,
    batch_paper_key_list: list,
    batch_bib: str,
    config: dict,
) -> bool:
    for downloaded_paper_title, batch_paper_title in zip(downloaded_paper_title_list, batch_paper_title_list):
        all_paper_title = downloaded_paper_title + batch_paper_title
        if (titles[0] in all_paper_title) or (titles[1] in all_paper_title) or (titles[2] in all_paper_title):
            logging.info("duplicate paper detected: %s", titles)
            return True

    all_paper_id_list = downloaded_paper_id_list + batch_paper_paper_id_list
    all_paper_key_list = downloaded_paper_key_list + batch_paper_key_list
    if (paper_id in all_paper_id_list) or (key_id in all_paper_key_list):
        logging.info("duplicate paper detected: %s", titles)
        return True

    all_bib = batch_bib
    for key in downloaded_paper_information.keys():
        all_bib += f"\n\n{downloaded_paper_information[key]}"

    questions = [
        (
            "# Please analyze and compare the candidate reference and the existing references in detail to "
            "determine whether the candidate reference exists among the existing references based on the "
            "following two conditions(If any condition is met, it is considered to exist):\n"
            "- They are the same paper (even if some information differs slightly)\n"
            "- They are different versions of the same paper (e.g., preprint vs. published version, etc., "
            "even if some information differs slightly)"
        ).strip()
    ]

    info_bench_eval_agent = TextJudgeAgent(config["judge_model_name"])
    judge_result = info_bench_eval_agent.evaluate(
        f"- The existing reference:\n```bib\n{all_bib}\n```\n\n\n- The candidate reference:\n```bib\n{bib}\n```",
        questions,
    )

    if all(judge_result["scores"]):
        logging.info("duplicate paper detected: %s", titles)
        return True

    return False
