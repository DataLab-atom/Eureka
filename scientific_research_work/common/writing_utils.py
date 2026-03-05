from __future__ import annotations

import json
from typing import Iterable, List, Sequence, Tuple

import bibtexparser


def build_writing_requirement(our_method_path: str, research_information_path: str) -> str:
    with open(our_method_path, "r", encoding="utf-8") as f:
        our_method_code = f.read()

    with open(research_information_path, "r", encoding="utf-8") as f:
        research_information = json.load(f)

    main_research = research_information.get("main_research", "")
    method_name = research_information.get("method_name", "")

    return (
        f"{main_research}.For this, We have proposed a method:**{method_name}**,"
        f" the python code implementation of our method is as follows:\n```\n{our_method_code}\n```\n"
    )


def text_image_separate(refer_data: Sequence) -> Tuple[List, List]:
    text_data: List = []
    image_data: List = []

    for data in refer_data:
        if getattr(data, "metadata", None) and data.metadata.get("image_base64"):
            image_data.append(data)
        else:
            text_data.append(data)

    return text_data, image_data


def format_reference(refer_data: Sequence) -> Tuple[List[dict], List[dict], str]:
    format_text_reference: List[dict] = []
    format_image_reference: List[dict] = []

    text_data_list, image_data_list = text_image_separate(refer_data)

    for text_data in text_data_list:
        format_text_reference.append(
            {
                "reference content": text_data.metadata.get("document_content", ""),
                "Bib citation": text_data.metadata.get("citation", ""),
            }
        )

    for image_data in image_data_list:
        format_image_reference.append(
            {
                "reference content": {
                    "image_base64": image_data.metadata.get("image_base64", ""),
                    "image_path": image_data.metadata.get("image_path", ""),
                },
                "Bib citation": image_data.metadata.get("image_references", ""),
            }
        )

    bib_content_list: List[str] = []
    for data in refer_data:
        metadata = getattr(data, "metadata", {}) or {}
        bib = metadata.get("citation") or metadata.get("image_references")
        if not bib:
            continue
        if bib not in bib_content_list:
            bib_content_list.append(bib)

    return format_text_reference, format_image_reference, "\n\n".join(bib_content_list)


def _extract_bib_title(bib_content: str) -> str | None:
    try:
        parser = bibtexparser.bparser.BibTexParser()
        bib_db = bibtexparser.loads(bib_content, parser=parser)
        if bib_db.entries:
            return bib_db.entries[0].get("title")
    except Exception:
        return None
    return None


def check_repeated_cite(
    refer_data: Iterable,
    all_paper_title_list: Sequence[str],
    current_used_paper_title: set[str],
) -> List:
    filtered = []
    used_titles = set(all_paper_title_list)

    for data in refer_data:
        metadata = getattr(data, "metadata", {}) or {}
        bib = metadata.get("citation") or metadata.get("image_references") or ""
        title = _extract_bib_title(bib) or bib
        if title in used_titles or title in current_used_paper_title:
            continue
        current_used_paper_title.add(title)
        filtered.append(data)

    return filtered
