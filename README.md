Building a vector database for PubMed
=====================================

# Pre-requisite
The project is built in python 3.10, and project dependency needs to be installed 

Create a new Python virtual environment
```console
python -m venv venv;
source venv/bin/activate;
```

Install the packages in requirements.txt
```console
pip install -r requirements.txt
```

# Download the PubMed xml.gz files

This first step is still manual. Download all files in https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ and https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/.

# Convert PubMed files to SQLite

The second step loads the PubMed data into a SQLite database. This implementation follows two principles:

1. Only one record per article. This means multiple items per abstract are combined into a single string. For example, there is a single string for the entire abstract, and one string for all co-authors.
2. Only information for RAG are included. This includes 
    - information for embedding (title, abstract, MeSH terms, keywords, and chemicals), and 
    - information for creating literature references (authors, journal, volume, issue, pagination, year).

The XML files are processed in order. This script can also be used to update the SQLite database when new XML files are published by NLM.

**Important**: XML files must be processed in order for the database to be valid. So if the script is executed on new XML files, these files must be newer (have higher numbers) than the previously processed XML files. If you must process an older file, you will need to then process all files that followed it to have the correct state.

To run, modify the `PubMedXmlToSqlite.yaml` file and run:
```python
PYTHONPATH=./: python PubMedSqliteIterator.py PubMedXmlToSqlite.yaml
```

# Convert SQLite to embedding vectors

The third step loads the data from the SQLite database and converts it to embedding vectors in Parquet files. We currently use an open-source embedding model, which you can specify in the yaml file.

To run, modify the `SqliteToEmbeddingVectors.yaml` file and run:
```python
PYTHONPATH=./: python SqliteToEmbeddingVectors.py SqliteToEmbeddingVectors.yaml
```

# Load the vectors in a vector database

The fourth step loads the embedding vectors from the Parquet files and inserts them into a PostgreSQL database with the [`pgvector` extension](https://github.com/pgvector/pgvector).

To run, modify the `LoadVectorsInStore.yaml` file and run:
```python
PYTHONPATH=./: python LoadVectorsInStore.py LoadVectorsInStore.yaml
```
The `store_type` argument in the YAML file can be set to `pgvector` for full precision, or `pgvector_halfvec` for half precision. 

Once the vectors are loaded, an index needs to be created. This requires a lot of memory. If the process runs out of memory, it will continue but at an extremely slow pace. In my experience, for 384-dimensional vectors at full precision, or for 768-dimensional vectors are half precision, the process requires up to 80GB of RAM. I found the best performacne using these settings:

```sql
SET maintenance_work_mem = '90GB'
SET max_parallel_maintenance_workers = 0
CREATE INDEX ON pubmed.vectors_snowflake_arctic_s USING hnsw (embedding vector_cosine_ops)
```
For half-precision the last line should read:

```sql
CREATE INDEX ON pubmed.vectors_snowflake_arctic_m USING hnsw (embedding halfvec_cosine_ops)
```




## License

RagPlayground is licensed under Apache License 2.0.

## Development status

Under development. Do not use.