import logging
import os
import sys
from typing import List

import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from numpy import ndarray

from PubMedSqliteIterator import fetch_pubmed_abstracts_for_embedding
from TransformerEmbedder import TransformerEmbedder
from SqliteToEmbeddingVectorsSettings import SqliteToEmbeddingVectorsSettings
from Logging import open_log

def store_in_parquet(pmids: List[int],
                     embeddings: ndarray,
                     publication_dates: List[int],
                     file_name: str):
    pmid_array = pa.array(pmids)
    pub_date_array = pa.array(publication_dates)
    embedding_arrays = [pa.array(embeddings[:, i]) for i in range(embeddings.shape[1])]

    table = pa.Table.from_arrays(
        arrays=[pmid_array, pub_date_array] + embedding_arrays,
        names=["pmid", "pub_date"] + [f"embedding_{i}" for i in range(embeddings.shape[1])]
    )
    pq.write_table(table, file_name)

def main(args: List[str]):
    with open(args[0]) as file:
        config = yaml.safe_load(file)
    settings = SqliteToEmbeddingVectorsSettings(config)
    open_log(settings.log_path)
    embedder = TransformerEmbedder(model_name=settings.embedding_model,
                                   embed_document_prompt=settings.embed_document_prompt,
                                   embed_query_prompt=settings.embed_query_prompt,
                                   embedding_batch_size=settings.embedding_batch_size)

    os.makedirs(settings.parquet_folder, exist_ok=True)

    total_count = 0

    for records in fetch_pubmed_abstracts_for_embedding(settings.sqlite_path, settings.batch_size):
        file_name = f"EmbeddingVectors{total_count + 1}_{total_count + len(records)}.parquet"
        file_name = os.path.join(settings.parquet_folder, file_name)
        if not os.path.isfile(file_name):
            logging.info(f"- Processing records {total_count + 1} to {total_count + len(records)}")

            logging.info("  Embedding")
            abstracts = [record[1] for record in records]
            embeddings = embedder.embed_documents(abstracts)

            logging.info("  Storing in Parquet")
            pmids = [int(record[0]) for record in records]
            publication_dates = [record[2] for record in records]

            store_in_parquet(pmids=pmids,
                             embeddings=embeddings,
                             publication_dates=publication_dates,
                             file_name=file_name)

        total_count = total_count + len(records)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to yaml file as argument")
    else:
        main(sys.argv[1:])
