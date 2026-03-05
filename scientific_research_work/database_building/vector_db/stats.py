import logging


def count_vector_database(vectordb) -> None:
    all_data = vectordb.get()
    data_count = len(all_data["ids"])
    logging.info("vector database contains %s records", data_count)

    paper_bib = set()
    for metadata in all_data["metadatas"]:
        if "image_references" in metadata:
            paper_bib.add(metadata["image_references"])
        else:
            paper_bib.add(metadata["citation"])

    logging.info("vector database contains %s papers", len(paper_bib))
