import logging

import pyarrow as pa
import pyarrow.parquet as pq

from PubMedIterator import fetch_pubmed_abstracts

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt",
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger.info("Started")


def write_to_parquet(records, file_name):
    # Define the schema
    schema = pa.schema([
        ('id', pa.int32()),
        ('abstract', pa.string()),
        ('publication_date', pa.date32())
    ])

    # Convert the records to a PyArrow Table
    table = pa.Table.from_pydict({
        'id': [record[0] for record in records],
        'abstract': [record[1] for record in records],
        'publication_date': [record[2] for record in records]
    }, schema=schema)

    # Write the table to a Parquet file
    pq.write_table(table, file_name)


def export_pubmed_to_parquet(output_dir, batch_size=100000):
    for i, records in enumerate(fetch_pubmed_abstracts(batch_size)):
        file_name = f"{output_dir}/pubmed_batch_{i + 1}.parquet"
        write_to_parquet(records, file_name)


if __name__ == "__main__":
    output_dir = "./PubmedParquet"

    export_pubmed_to_parquet(output_dir)
