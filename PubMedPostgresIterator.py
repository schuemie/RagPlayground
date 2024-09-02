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
    # DROP TABLE IF EXISTS medline.mesh_text;
    # DROP INDEX IF EXISTS idx_mesh_text;
    #
    # CREATE TABLE medline.mesh_text AS
    # SELECT pmid,
    #     pmid_version,
    #     string_agg(descriptorname, '\n') AS mesh_terms
    # FROM medline.medcit_meshheadinglist_meshheading
    # GROUP BY pmid,
    #     pmid_version;
    #
    # CREATE INDEX idx_mesh_text ON medline.mesh_text (pmid, pmid_version);
    # """

    # sql = """
    # DROP TABLE IF EXISTS medline.abstract_text;
    # DROP INDEX IF EXISTS idx_abstract_text;
    #
    # CREATE TABLE medline.abstract_text AS
    # SELECT pmid,
    #     pmid_version,
    #     string_agg(value, '\n') AS abstract_text
    # FROM medline.medcit_art_abstract_abstracttext
    # GROUP BY pmid,
    #     pmid_version;
    #
    # CREATE INDEX idx_abstract_text ON medline.abstract_text (pmid, pmid_version);
    # """

   #  sql = """
   #  DROP TABLE IF EXISTS medline.med_cit_pd;
   #  DROP INDEX IF EXISTS med_cit_pd;
   #
   #  CREATE TABLE medline.med_cit_pd AS
   #  SELECT pmid,
   #      pmid_version,
   #      art_arttitle,
   #      pmid_to_date.date AS publication_date
   # FROM medline.medcit
   # INNER JOIN medline.pmid_to_date
   #      ON medcit.pmid = pmid_to_date.pmid
   #          AND medcit.pmid_version = pmid_to_date.pmid_version;
   #
   #  CREATE INDEX idx_med_cit_pd ON medline.med_cit_pd (pmid, pmid_version);
   #  """


    # sql = """
    # SELECT medcit.pmid,
    #     CONCAT(art_arttitle,
    #            '\n',
    #            abstract_text,
    #            '\n',
    #            mesh_terms
    #     ) AS text,
    #     pmid_to_date.date AS publication_date
    # FROM medline.medcit
    # LEFT JOIN (
	# 	SELECT pmid,
	# 		pmid_version,
	# 		string_agg(value, ', ') AS abstract_text
	# 	FROM medline.medcit_art_abstract_abstracttext
	# 	GROUP BY pmid,
	# 		pmid_version
	# 	) abstract
    #     ON medcit.pmid = abstract.pmid
    #         AND medcit.pmid_version = abstract.pmid_version
    # LEFT JOIN (
	# 	SELECT pmid,
	# 		pmid_version,
	# 		string_agg(descriptorname, ', ') AS mesh_terms
	# 	FROM medline.medcit_meshheadinglist_meshheading
	# 	GROUP BY pmid,
	# 		pmid_version
	# 	) mesh
    #     ON medcit.pmid = mesh.pmid
    #         AND medcit.pmid_version = mesh.pmid_version
    # INNER JOIN medline.pmid_to_date
    #     ON medcit.pmid = pmid_to_date.pmid
    #         AND medcit.pmid_version = pmid_to_date.pmid_version
	# WHERE medcit.pmid_version = 1;
    # """
    sql = """
    SELECT med_cit_pd.pmid,
        CONCAT(art_arttitle, 
               '\n', 
               abstract_text, 
               '\n', 
               mesh_terms
        ) AS text,
        publication_date
    FROM medline.med_cit_pd  
    LEFT JOIN medline.abstract_text
         ON med_cit_pd.pmid = abstract_text.pmid
            AND med_cit_pd.pmid_version = abstract_text.pmid_version
    LEFT JOIN medline.mesh_text
        ON med_cit_pd.pmid = mesh_text.pmid
            AND med_cit_pd.pmid_version = mesh_text.pmid_version        
    WHERE med_cit_pd.pmid_version = 1;  	
    """
    # sql = """
    # SELECT pmid, text, publication_date FROM scratch.temp_pubmed;
    # """
    cursor.execute(sql)

    while True:
        records = cursor.fetchmany(batch_size)
        if not records:
            break
        yield records

    cursor.close()
    conn.close()
