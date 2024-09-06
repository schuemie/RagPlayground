import logging
import os
import sys
from typing import List

import numpy as np
import pyarrow.parquet as pq
import yaml

from tqdm import tqdm
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

# from HNSWIndex import HNSWIndex
from LoadVectorsInStoreSettings import LoadVectorsInStoreSettings
from Logging import open_log

load_dotenv()

def load_vectors_in_hnswlib(settings: LoadVectorsInStoreSettings):
    pass
    # index = HNSWIndex(dim=384,
    #                   max_elements=40000000,
    #                   index_path=settings.path)
    # file_list = sorted([f for f in os.listdir(settings.parquet_folder) if f.endswith(".parquet")])
    # for i in tqdm(range(0, len(file_list))):
    #     file_name = file_list[i]
    #     logging.info(f"Processing Parquet file '{file_name}'")
    #     file_path = os.path.join(settings.parquet_folder, file_name)
    #     parquet_file = pq.ParquetFile(file_path)
    #     for row_group_idx in range(parquet_file.num_row_groups):
    #         row_group = parquet_file.read_row_group(row_group_idx)
    #         pmids = row_group.column("pmid").to_numpy()
    #         # pub_dates = row_group.column("pub_date").to_numpy()
    #         embedding_columns = [row_group.column(i).to_numpy() for i in range(2, row_group.num_columns)]
    #         vectors = np.column_stack(embedding_columns)
    #         logging.info(f"- Adding {len(vectors)} vectors to the index.")
    #         index.add_vectors(vectors, pmids)
    # logging.info(f"Index size is now {index.get_current_count()} records")
    # logging.info(f"Saving index to {settings.path}")
    # index.save_index(settings.path)

def load_vectors_in_pgvector(settings: LoadVectorsInStoreSettings):
    if os.getenv("POSTGRES_SERVER") is None:
        raise Exception("Must set environmental variables POSTGRES_SERVER, POSTGRES_USER, POSTGRES_PASSWORD, and "
                        "POSTGRES_DATABASE when writing to Postgres.")
    conn = psycopg.connect(host=os.getenv("POSTGRES_SERVER"),
                           user=os.getenv("POSTGRES_USER"),
                           password=os.getenv("POSTGRES_PASSWORD"),
                           dbname=os.getenv("POSTGRES_DATABASE"),
                           autocommit=True)
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)

    # create table
    statement = sql.SQL("CREATE TABLE IF NOT EXISTS {schema}.{table} (pmid INT PRIMARY KEY, embedding vector({dimensions}))").format(
        schema=sql.Identifier(settings.schema),
        table=sql.Identifier(settings.table),
        dimensions=sql.Literal(settings.dimensions)
    )
    conn.execute(statement)

    cur = conn.cursor()
    statement = sql.SQL("COPY {schema}.{table} (pmid, embedding) FROM STDIN WITH (FORMAT BINARY)").format(
        schema=sql.Identifier(settings.schema),
        table=sql.Identifier(settings.table)
    )
    with cur.copy(statement) as copy:
        copy.set_types(["int4", "vector"])

        # Iterate over Parquet files:
        total_count = 0
        file_list = sorted([f for f in os.listdir(settings.parquet_folder) if f.endswith(".parquet")])
        for i in tqdm(range(0, len(file_list))):
            file_name = file_list[i]
            logging.info(f"Processing Parquet file '{file_name}'")
            file_path = os.path.join(settings.parquet_folder, file_name)
            parquet_file = pq.ParquetFile(file_path)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_idx)
                pmids = row_group.column("pmid").to_pylist()
                embedding_columns = [row_group.column(i).to_pylist() for i in range(2, row_group.num_columns)]
                logging.info(f"- Inserting {len(pmids)} vectors")
                # Iterate over rows
                for i, embedding in enumerate(zip(*embedding_columns)):
                    pmid = int(pmids[i])
                    copy.write_row([pmid, embedding])
                total_count = total_count + len(pmids)
                logging.info(f"- Inserted {total_count} vectors in total")
        # Flush data
        while conn.pgconn.flush() == 1:
            pass

    query = sql.SQL("SELECT COUNT(*) FROM {schema}.{table}").format(
        schema=sql.Identifier(settings.schema),
        table=sql.Identifier(settings.table)
    )
    result = cur.execute(query)
    count = result.fetchone()[0]
    logging.info(f"Index size is now {count} records")



def main(args: List[str]):
    with open(args[0]) as file:
        config = yaml.safe_load(file)
    settings = LoadVectorsInStoreSettings(config)
    open_log(settings.log_path)

    if settings.store_type ==settings.HSNWLIB:
        load_vectors_in_hnswlib(settings)
    else:
        load_vectors_in_pgvector(settings)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to yaml file as argument")
    else:
        main(sys.argv[1:])

# CREATE INDEX ON pubmed.vectors_snowflake_arctic_s USING hnsw (embedding vector_cosine_ops)
# ANALYZE pubmed.vectors_snowflake_arctic_s