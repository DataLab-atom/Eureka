import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from llm.tokens import num_tokens_from_string


def split_text_with_headings(
    text: str,
    reference: dict,
    min_tokens: int,
    max_tokens: int,
    chunk_size: int,
    embedding_model_name: str,
) -> list[Document]:
    heading_pattern = re.compile(r"(\n#+ .*\n)")
    marked_text = heading_pattern.sub(r"|||\1|||", text)

    segments = []
    current_segment = ""
    temp_parts = marked_text.split("|||")

    markdown_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    for part in temp_parts:
        if heading_pattern.match(part):
            if num_tokens_from_string.get_string_num_tokens(current_segment, embedding_model_name) > max_tokens:
                split_docs = markdown_splitter.split_text(current_segment.strip())
                chunks = ""
                for idx, chunk in enumerate(split_docs):
                    chunk_tokens = num_tokens_from_string.get_string_num_tokens(chunks, embedding_model_name)
                    if min_tokens <= chunk_tokens < max_tokens:
                        segments.append(
                            Document(
                                page_content=chunks.strip(),
                                metadata={"document_content": chunks.strip(), "citation": reference},
                            )
                        )
                        chunks = f"{split_docs[idx - 1]}\n{chunk}"
                    else:
                        chunks += chunk + "\n"
                segments.append(
                    Document(
                        page_content=chunks.strip(),
                        metadata={"document_content": chunks.strip(), "citation": reference},
                    )
                )
                current_segment = part
            else:
                current_tokens = num_tokens_from_string.get_string_num_tokens(current_segment, embedding_model_name)
                if chunk_size <= current_tokens < max_tokens:
                    segments.append(
                        Document(
                            page_content=current_segment,
                            metadata={"document_content": current_segment, "citation": reference},
                        )
                    )
                    current_segment = part
                else:
                    current_segment += part
        else:
            current_segment += part

    if current_segment:
        if num_tokens_from_string.get_string_num_tokens(current_segment, embedding_model_name) > max_tokens:
            split_docs = markdown_splitter.split_text(current_segment.strip())
            chunks = ""
            for idx, chunk in enumerate(split_docs):
                chunk_tokens = num_tokens_from_string.get_string_num_tokens(chunks, embedding_model_name)
                if min_tokens <= chunk_tokens < max_tokens:
                    segments.append(
                        Document(
                            page_content=chunks.strip(),
                            metadata={"document_content": chunks.strip(), "citation": reference},
                        )
                    )
                    chunks = f"{split_docs[idx - 1]}\n{chunk}"
                else:
                    chunks += chunk + "\n"
            segments.append(
                Document(
                    page_content=chunks.strip(),
                    metadata={"document_content": chunks.strip(), "citation": reference},
                )
            )
        else:
            segments.append(
                Document(
                    page_content=current_segment.strip(),
                    metadata={"document_content": current_segment.strip(), "citation": reference},
                )
            )

    return segments
