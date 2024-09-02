from datetime import date
from typing import List, Any

import chromadb
from numpy import ndarray

DAYS_OFFSET = date(1970, 1, 1).toordinal()


def store_in_chromadb(ids: List[Any],
                      embeddings: ndarray,
                      publication_dates: List[int],
                      folder: str = "E:/Medline/PubMedChromaDb") -> None:
    client = chromadb.PersistentClient(path=folder)
    collection = client.get_or_create_collection(name="pubmed_abstracts")

    ids = [str(id) for id in ids]
    meta_datas = [{'publication_date': publication_date} for publication_date in publication_dates]

    batch_size = 166  # Max batch size allowed by collection.add()
    for i in range(0, len(ids), batch_size):
        collection.add(ids=ids[i:i + batch_size],
                       embeddings=embeddings[i:i + batch_size],
                       metadatas=meta_datas[i:i + batch_size])