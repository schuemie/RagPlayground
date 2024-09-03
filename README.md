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

Not yet finished. ChromaDB does not appear to be able to handle this size.

## License

RagPlayground is licensed under Apache License 2.0.

## Development status

Under development. Do not use.