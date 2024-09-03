import os
import sqlite3


def fetch_pubmed_abstracts_for_embedding(batch_size: int = 100000):
    """
    An iterator that fetches PubMed abstracts in batches. The contents are aimed at creating embedding vectors for
    retrieval.

    :param batch_size: The size of the batches the iterator returns
    :return: A tuple of 3: pmids, texts, and publication dates (toordinal integers), each of length batch_size.
    """
    connection = sqlite3.connect('PubMed.sqlite')
    cursor = connection.cursor()

    sql = """
    SELECT pmid,
        CASE WHEN title IS NULL THEN '' ELSE title || '\n\n' END ||
            CASE WHEN abstract IS NULL THEN '' ELSE abstract || '\n\n' END ||
            CASE WHEN mesh_terms IS NULL THEN '' ELSE 'MeSH terms:\n' || mesh_terms || '\n\n' END ||
            CASE WHEN keywords IS NULL THEN '' ELSE 'Keywords:\n' || keywords || '\n\n' END ||
            CASE WHEN chemicals IS NULL THEN '' ELSE 'Chemicals:\n' || chemicals || '\n\n' END AS text,
        publication_date
    FROM pubmed_articles;  	
    """
    # cursor.execute(sql)
    # x = cursor.fetchone()
    # x
    while True:
        records = cursor.fetchmany(batch_size)
        if not records:
            break
        yield records

    cursor.close()
    connection.close()
