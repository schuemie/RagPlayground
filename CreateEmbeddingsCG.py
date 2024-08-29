import logging
import os
from datetime import date

import psycopg2
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import numpy as np

load_dotenv()

DAYS_OFFSET = date(1970, 1, 1).toordinal()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("Started")


# Step 1: Connect to PostgreSQL Database and Fetch Data
def fetch_pubmed_abstracts(batch_size: int):
    conn = psycopg2.connect(host=os.getenv("SERVER"),
                            user=os.getenv("USER"),
                            password=os.getenv("PASSWORD"),
                            dbname=os.getenv("DATABASE"))
    cursor = conn.cursor()

    # sql = """
    # SELECT medcit.pmid,
    #     CONCAT(art_arttitle,
    #            '\n',
    #            string_agg(abstract.value, '\n'),
    #            '\n',
    #            string_agg(mesh.descriptorname, ', ')
    #     ) AS text,
    #     pmid_to_date.date AS publication_date
    # FROM medline.medcit
    # INNER JOIN medline.medcit_art_abstract_abstracttext abstract
    #     ON medcit.pmid = abstract.pmid
    #         AND medcit.pmid_version = abstract.pmid_version
    # INNER JOIN medline.medcit_meshheadinglist_meshheading mesh
    #     ON medcit.pmid = mesh.pmid
    #         AND medcit.pmid_version = mesh.pmid_version
    # INNER JOIN medline.pmid_to_date
    #     ON medcit.pmid = pmid_to_date.pmid
    #         AND medcit.pmid_version = pmid_to_date.pmid_version
    # WHERE medcit.pmid_version = 1
    # GROUP BY medcit.pmid,
    #     medcit.pmid_version,
    #     art_arttitle,
    #     pmid_to_date.date
    # LIMIT 10000;
    # """
    sql = """
    SELECT pmid, text, publication_date FROM scratch.temp_pubmed;
    """
    cursor.execute(sql)

    while True:
        records = cursor.fetchmany(batch_size)
        if not records:
            break
        yield records

    cursor.close()
    conn.close()


# Step 2: Generate Embeddings using HuggingFace Transformers
class TransformerEmbedder:
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1.5'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def generate_embeddings(self, texts, batch_size: int):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = ["search_document: " + text for text in texts[i:i + batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Get the attention mask and convert it to float
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

                # Mask the hidden states
                masked_hidden_states = outputs.last_hidden_state * attention_mask

                # Calculate the sum of the hidden states and the sum of the attention mask
                summed_hidden_states = masked_hidden_states.sum(dim=1)
                summed_mask = attention_mask.sum(dim=1)

                # Calculate the mean by dividing the summed hidden states by the summed mask
                batch_embeddings = (summed_hidden_states / summed_mask).cpu().numpy()
                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


# Step 3: Store Embeddings in ChromaDB without Text
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
