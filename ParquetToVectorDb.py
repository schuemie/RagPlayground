import logging
import os

import pyarrow.parquet as pq

from TransformerEmbedder import TransformerEmbedder
from ChromaDb import store_in_chromadb


logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("Started")


def process_parquet_files(folder: str):
    embedder = TransformerEmbedder(batch_size=32)
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
        store_in_chromadb(ids=ids,
                          embeddings=embeddings,
                          publication_dates=publication_dates)

        total_count = total_count + len(ids)


if __name__ == "__main__":
    process_parquet_files("./PubmedParquet")
