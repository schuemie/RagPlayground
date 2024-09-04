import logging
import os
import sys
from typing import List

import numpy as np
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from HNSWIndex import HNSWIndex
from LoadVectorsInStoreSettings import LoadVectorsInStoreSettings
from Logging import open_log


def main(args: List[str]):
    with open(args[0]) as file:
        config = yaml.safe_load(file)
    settings = LoadVectorsInStoreSettings(config)
    open_log(settings.log_path)

    index = HNSWIndex(dim=384,
                      max_elements=40000000,
                      index_path=settings.vector_store_path)
    file_list = sorted([f for f in os.listdir(settings.parquet_folder) if f.endswith(".parquet")])
    for i in tqdm(range(0, len(file_list))):
        file_name = file_list[i]
        logging.info(f"Processing Parquet file '{file_name}'")
        file_path = os.path.join(settings.parquet_folder, file_name)
        parquet_file = pq.ParquetFile(file_path)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            pmids = row_group.column("pmid").to_numpy()  # Or to_pylist() to get Python lists
            # pub_dates = row_group.column("pub_date").to_numpy()
            embedding_columns = [row_group.column(i).to_numpy() for i in range(2, row_group.num_columns)]
            vectors = np.column_stack(embedding_columns)
            logging.info(f"- Adding {len(vectors)} vectors to the index.")
            index.add_vectors(vectors, pmids)
    logging.info(f"Index size is now {index.get_current_count()} records")
    logging.info(f"Saving index to {settings.vector_store_path}")
    index.save_index(settings.vector_store_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Must provide path to yaml file as argument")
    else:
        main(sys.argv[1:])