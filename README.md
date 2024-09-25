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


## Creating the vector index
Once the vectors are loaded, an index needs to be created. This requires a lot of memory. It is probably best to partition the table, because then a separate index will be created for each partition.

In my experience, 37 million v768-dimensional vectors stored as `halfvec` will require about 80GB of memory in total. We could create a new table with 8 partitions, and copy the data from the staging table:

```sql
CREATE TABLE pubmed.vectors_snowflake_arctic_m_partitioned (
    pmid INT PRIMARY KEY,
    embedding halfvec(768)
) PARTITION BY HASH(pmid);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_1 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 0);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_2 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 1);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_3 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 2);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_4 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 3);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_5 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 4);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_6 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 5);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_7 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 6);

CREATE TABLE vectors_snowflake_arctic_m_partitioned_8 PARTITION OF vectors_snowflake_arctic_m_partitioned
    FOR VALUES WITH (MODULUS 8, REMAINDER 7);

INSERT INTO vectors_snowflake_arctic_m_partitioned
SELECT * FROM vectors_snowflake_arctic_m;
```
Because we now have 8 partitions, we'll need about 80 / 8 = 10GB of memory to create the index:
```sql
SET maintenance_work_mem = '10GB'
SET max_parallel_maintenance_workers = 4
CREATE INDEX ON pubmed.vectors_snowflake_arctic_m_partitioned USING hnsw (embedding halfvec_cosine_ops)
```

## License

RagPlayground is licensed under Apache License 2.0.

## Development status

Under development. Do not use.