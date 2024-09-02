import os
import sqlite3


def fetch_pubmed_abstracts(batch_size: int = 100000):
    conn = sqlite3.connect('PubMed.sqlite')
    cursor = conn.cursor()

    sql = """
    SELECT pmid,
    CASE WHEN title IS NULL THEN '' ELSE title || '\n' END ||
        CASE WHEN abstract IS NULL THEN '' ELSE abstract || '\n' END ||
        CASE WHEN mesh_terms IS NULL THEN '' ELSE mesh_terms END AS text,
    mesh_terms,    
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
    conn.close()
