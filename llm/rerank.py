import logging
from typing import List

import requests
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document

from llm.env import get_env


def _rerank_url() -> str:
    return get_env("RERANK_URL", "https://api.siliconflow.cn/v1/rerank") or ""


def _rerank_key() -> str:
    return get_env("RERANK_RETRIEVE_KEY", "") or ""


def get_reranker_result(model_name: str, query: str, documents: List[str], n: int):
    payload = {
        "model": model_name,
        "query": query,
        "documents": documents,
        "top_n": n,
    }
    headers = {
        "Authorization": f"Bearer {_rerank_key()}",
        "Content-Type": "application/json",
    }

    response = requests.post(_rerank_url(), json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["results"]


def reranking_intercept(
    query: str,
    data: List[Document],
    k: int,
    reranker_path: str,
    device: str,
    reranker_model_source: str,
) -> List[Document]:
    if not data:
        return []
    if k <= 0:
        return []
    if k > len(data):
        k = len(data)

    if reranker_model_source == "local":
        reranking_model = FlagReranker(reranker_path, devices=device)
        scores = reranking_model.compute_score([(query, doc.page_content) for doc in data])
        reranked_data = [doc for _, doc in sorted(zip(scores, data), reverse=True)]
        return reranked_data[:k]

    text_content = [doc.page_content for doc in data]
    try:
        scores = get_reranker_result(reranker_path, query, text_content, k)
        return [data[score["index"]] for score in scores]
    except Exception as exc:
        logging.warning("reranker API failed, fallback to original order: %s", exc)
        return data[:k]
