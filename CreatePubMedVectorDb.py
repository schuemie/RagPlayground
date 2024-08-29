import logging
from datetime import date

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from PubMedIterator import fetch_pubmed_abstracts
from TransformerEmbedder import TransformerEmbedder

DAYS_OFFSET = date(1970, 1, 1).toordinal()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("Started")


def store_in_chromadb(records, embeddings):
    client = chromadb.PersistentClient(path = "./PubMedChromaDb")
    collection = client.get_or_create_collection(name="pubmed_abstracts", embedding_function=DefaultEmbeddingFunction())

    ids = [str(record[0]) for record in records]
    metadatas = [{'publication_date': (record[2].toordinal() - DAYS_OFFSET)} for record in records]
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)


# Main process
def process_pubmed_abstracts(batch_size=1000, embedding_batch_size=32):
    embedder = TransformerEmbedder()
    total_count = 0

    for records in fetch_pubmed_abstracts(batch_size):
        logging.info(f"- Processing records {total_count} to {total_count + len(records) - 1}")

        logging.info("  Embedding")
        abstracts = [record[1] for record in records]
        embeddings = embedder.generate_embeddings(abstracts, batch_size=embedding_batch_size)

        logging.info("  Storing in ChromaDB")
        store_in_chromadb(records, embeddings)

        total_count = total_count + len(records)


if __name__ == "__main__":
    process_pubmed_abstracts()
