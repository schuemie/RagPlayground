import os
import sqlite3


def fetch_pubmed_abstracts(batch_size: int = 100000):
    conn = sqlite3.connect('PubMed.sqlite')
    cursor = conn.cursor()

    sql = """
    SELECT pmid,
        title || '\n' || abstract || '\n' ||  mesh_terms AS text,
        publication_date
    FROM pubmed_articles
    LIMIT -1 OFFSET 1560000;  	
    """
    cursor.execute(sql)

    while True:
        records = cursor.fetchmany(batch_size)
        if not records:
            break
        yield records

    cursor.close()
    conn.close()
