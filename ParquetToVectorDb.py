import logging
import os
from datetime import date

import pyarrow.parquet as pq
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from TransformerEmbedder import TransformerEmbedder

DAYS_OFFSET = date(1970, 1, 1).toordinal()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("Started")


def store_in_chromadb(ids, publication_dates, embeddings):
    client = chromadb.PersistentClient(path = "./PubMedChromaDb")
    collection = client.get_or_create_collection(name="pubmed_abstracts", embedding_function=DefaultEmbeddingFunction())

    ids = [str(id) for id in ids]
    metadatas = [{'publication_date': (publication_date.toordinal() - DAYS_OFFSET)} for publication_date in publication_dates]
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)


def process_parquet_files(folder):
    embedder = TransformerEmbedder()
    total_count = 0
    files = os.listdir(folder)
    for file_name in files:
        table = pq.read_table(os.path.join(folder, file_name))
        data = table.to_pydict()
        ids = data['id']
        abstracts = data['abstract']
        publication_dates = data['publication_date']
        logging.info(f"- Processing records {total_count} to {total_count + len(ids) - 1}")

        logging.info("  Embedding")
        embeddings = embedder.generate_embeddings(abstracts, batch_size=128)

        logging.info("  Storing in ChromaDB")
        store_in_chromadb(ids, publication_dates, embeddings)

        total_count = total_count + len(ids)


if __name__ == "__main__":
    process_parquet_files("./PubmedParquet")
