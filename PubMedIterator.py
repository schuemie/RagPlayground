import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def fetch_pubmed_abstracts(batch_size: int = 100000):
    conn = psycopg2.connect(host=os.getenv("SERVER"),
                            user=os.getenv("USER"),
                            password=os.getenv("PASSWORD"),
                            dbname=os.getenv("DATABASE"))
    cursor = conn.cursor()

    # sql = """
    # SELECT medcit.pmid,
    #     CONCAT(art_arttitle,
    #            '\n',
    #            string_agg(abstract.value, '\n'),
    #            '\n',
    #            string_agg(mesh.descriptorname, ', ')
    #     ) AS text,
    #     pmid_to_date.date AS publication_date
    # FROM medline.medcit
    # INNER JOIN medline.medcit_art_abstract_abstracttext abstract
    #     ON medcit.pmid = abstract.pmid
    #         AND medcit.pmid_version = abstract.pmid_version
    # INNER JOIN medline.medcit_meshheadinglist_meshheading mesh
    #     ON medcit.pmid = mesh.pmid
    #         AND medcit.pmid_version = mesh.pmid_version
    # INNER JOIN medline.pmid_to_date
    #     ON medcit.pmid = pmid_to_date.pmid
    #         AND medcit.pmid_version = pmid_to_date.pmid_version
    # WHERE medcit.pmid_version = 1
    # GROUP BY medcit.pmid,
    #     medcit.pmid_version,
    #     art_arttitle,
    #     pmid_to_date.date
    # LIMIT 10000;
    # """
    sql = """
    SELECT pmid, text, publication_date FROM scratch.temp_pubmed;
    """
    cursor.execute(sql)

    while True:
        records = cursor.fetchmany(batch_size)
        if not records:
            break
        yield records

    cursor.close()
    conn.close()
