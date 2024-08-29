import os
import logging

import psycopg2
from dotenv import load_dotenv
from safetensors import torch
from transformers import AutoTokenizer, AutoModel

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="log.txt", level=logging.INFO)
logger.info("Started")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
model.eval()


connection = psycopg2.connect(host=os.getenv("SERVER"),
                              user=os.getenv("USER"),
                              password=os.getenv("PASSWORD"),
                              dbname=os.getenv("DATABASE"))


with connection.cursor(name='fetch_large_result') as cursor:

    cursor.itersize = 100000
    sql = """
    SELECT medcit.pmid,
        medcit.pmid_version,
        art_arttitle AS title, 
        string_agg(abstract.value, '\n') AS abstract,
        string_agg(mesh.descriptorname, ', ') AS mesh_headers,
        pmid_to_date.date AS publication_date
    FROM medline.medcit
    INNER JOIN medline.medcit_art_abstract_abstracttext abstract
        ON medcit.pmid = abstract.pmid
            AND medcit.pmid_version = abstract.pmid_version
    INNER JOIN medline.medcit_meshheadinglist_meshheading mesh
        ON medcit.pmid = mesh.pmid
            AND medcit.pmid_version = mesh.pmid_version
    INNER JOIN medline.pmid_to_date
        ON medcit.pmid = pmid_to_date.pmid
            AND medcit.pmid_version = pmid_to_date.pmid_version  
    GROUP BY medcit.pmid,
        medcit.pmid_version,
        art_arttitle
    OFFSET 8000000;     
    """
    cursor.execute(sql)
    token_count = 0
    record_count = 0
    for row in cursor:
        if row[2] is None:
            title = ""
        else:
            title = row[2]
        if row[3] is None:
            abstract = ""
        else:
            abstract = row[3]
        if row[4] is None:
            mesh_headers = ""
        else:
            mesh_headers = row[4]
        text = title + "\n" + abstract + "\n" + mesh_headers
        token_count = token_count + len(text_tokens)
        record_count = record_count + 1
        if record_count % 100000 == 0:
            print(f"Records: {record_count}, tokens: {token_count}")
            logger.info(f"Records: {record_count}, tokens: {token_count}")
    print(f"Records: {record_count}, tokens: {token_count}")
    logger.info(f"Records: {record_count}, tokens: {token_count}")