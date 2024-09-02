import logging

from PubMedSqliteIterator import fetch_pubmed_abstracts
from TransformerEmbedder import TransformerEmbedder
from ChromaDb import store_in_chromadb

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("Started")


# Main process
def process_pubmed_abstracts(batch_size=10000):
    embedder = TransformerEmbedder()
    total_count = 0

    for records in fetch_pubmed_abstracts(batch_size):
        for record in records:
            if record[1] is None:
                print(record[0])
        logging.info(f"- Processing records {total_count} to {total_count + len(records) - 1}")
        #
        # logging.info("  Embedding")
        # abstracts = [record[1] for record in records]
        # embeddings = embedder.embed_documents(abstracts)
        #
        # logging.info("  Storing in ChromaDB")
        # ids = [record[0] for record in records]
        # publication_dates = [record[2] for record in records]
        # store_in_chromadb(ids=ids,
        #                   embeddings=embeddings,
        #                   publication_dates=publication_dates)

        total_count = total_count + len(records)


if __name__ == "__main__":
    process_pubmed_abstracts()
