import logging
import os
from datetime import date
from typing import List, Any

import pyarrow.parquet as pq
import chromadb
from numpy import ndarray

from TransformerEmbedder import TransformerEmbedder

DAYS_OFFSET = date(1970, 1, 1).toordinal()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("Started")


def store_in_chromadb(ids: List[Any], publication_dates: List[date], embeddings: ndarray) -> None:
    client = chromadb.PersistentClient(path="./PubMedChromaDb")
    collection = client.get_or_create_collection(name="pubmed_abstracts")

    ids = [str(id) for id in ids]
    meta_datas = [{'publication_date': (pd.toordinal() - DAYS_OFFSET)} for pd in publication_dates]

    batch_size = 166  # Max batch size allowed by collection.add()
    for i in range(0, len(ids), batch_size):
        collection.add(ids=ids[i:i + batch_size],
                       embeddings=embeddings[i:i + batch_size],
                       metadatas=meta_datas[i:i + batch_size])


def process_parquet_files(folder: str):
    embedder = TransformerEmbedder(batch_size=128)
    total_count = 0
    files = os.listdir(folder)
    for file_name in files:
        table = pq.read_table(os.path.join(folder, file_name))
        data = table.to_pydict()
        ids = data['id']
        abstracts = data['abstract']
        publication_dates = data['publication_date']

        # ids = ids[0:10000]
        # abstracts = abstracts[0:10000]
        # publication_dates = publication_dates[0:10000]

        logging.info(f"- Processing records {total_count} to {total_count + len(ids) - 1}")

        logging.info("  Embedding")
        embeddings = embedder.embed_documents(abstracts)

        logging.info("  Storing in ChromaDB")
        store_in_chromadb(ids, publication_dates, embeddings)

        total_count = total_count + len(ids)


if __name__ == "__main__":
    process_parquet_files("./PubmedParquet")
