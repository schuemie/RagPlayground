import csv
import json
import os
import random
import re
import pickle
from typing import List

import psycopg
from dotenv import load_dotenv

load_dotenv()

TREC_COVID_SOURCE_FOLDER = "/Users/schuemie/Downloads/trec-covid"
BIOASQ_SOURCE_FILE = "/Users/schuemie/Downloads/BioASQ-training12b/training12b_new.json"


def parse_trec_covid():
    # Load queries ----------------------------------------------
    with open(os.path.join(TREC_COVID_SOURCE_FOLDER, "queries.jsonl"), "r", encoding="utf-8") as f:
        queries = f.read().split("\n")
    id_pattern = r"\"_id\":\s\"([a-z0-9]+)\""
    query_pattern = r"\"text\":\s\"(.*?)\","
    query_id_to_query = {re.search(id_pattern, query).group(1): re.search(query_pattern, query).group(1) for query
                         in queries if re.search(id_pattern, query) is not None}

    # Load QRELS -----------------------------------------------
    with open(os.path.join(TREC_COVID_SOURCE_FOLDER, "qrels", "test.tsv")) as f:
        qrels = csv.reader(f, delimiter="\t")
        qrels = [qrel for qrel in qrels][1:]
    doc_ids_in_qrels = set([qrel[1] for qrel in qrels])

    # Load corpus -------------------------------------------------
    # Corpus file appears incorrect JSON, so hacking the parsing:
    with open(os.path.join(TREC_COVID_SOURCE_FOLDER, "corpus.jsonl"), "r", encoding="utf-8") as f:
        corpus = f.read().split("\n")
    pattern = r"\"_id\":\s\"([a-z0-9]+)\""
    doc_id_to_doc = {re.search(pattern, doc).group(1): doc for doc in corpus if re.search(pattern, doc) is not None}

    # Find PMIDs:
    doc_id_to_pmid = {item[0]: re.search(r"\"pubmed_id\":\s\"([0-9]+)\"", item[1]).group(1) for item in
                      doc_id_to_doc.items() if re.search(r"\"pubmed_id\":\s\"([0-9]+)\"", item[1]) is not None}

    # Try to find PMIDs for remainder based on titles:
    doc_id_to_title = {item[0]: re.search(r"\"title\":\s\"(.*?)\",", item[1]).group(1) for item in doc_id_to_doc.items()
                       if re.search(r"\"pubmed_id\":\s\"([0-9]+)\"", item[1]) is None}
    postgres_conn = psycopg.connect(host=os.getenv("POSTGRES_SERVER"),
                                    user=os.getenv("POSTGRES_USER"),
                                    password=os.getenv("POSTGRES_PASSWORD"),
                                    dbname=os.getenv("POSTGRES_DATABASE"))
    cursor = postgres_conn.cursor()
    cursor.execute(f"SET SEARCH_PATH={os.getenv('POSTGRES_SCHEMA')};")

    cursor.execute("BEGIN TRANSACTION;")
    cursor.execute("CREATE TEMPORARY TABLE doc_id_to_title (doc_id TEXT, title TEXT);")
    cursor.executemany("INSERT INTO doc_id_to_title (doc_id, title) VALUES (%s, %s)", doc_id_to_title.items())
    sql = """
    SELECT doc_id, 
        pmid 
    FROM doc_id_to_title 
    INNER JOIN pubmed_articles
        ON  doc_id_to_title.title = pubmed_articles.title
    """
    cursor.execute(sql)
    doc_id_to_pmid_2 = cursor.fetchall()
    doc_id_to_pmid.update({doc_id: pmid for doc_id, pmid in doc_id_to_pmid_2})

    # Convert query IDs to ints:
    query_id_to_int = {}
    for i, query_id in enumerate(query_id_to_query):
        query_id_to_int[query_id] = i

    # Combine objects and save:
    pmids = [int(pmid) for pmid in doc_id_to_pmid.values()]
    query_id_to_qrels = {}
    for query_id in query_id_to_query:
        pmid_to_score = {int(doc_id_to_pmid[doc_id]): int(score) for id, doc_id, score
                         in qrels if id == query_id and doc_id in doc_id_to_pmid}
        query_id_to_qrels[query_id_to_int[query_id]] = pmid_to_score
    query_id_to_query = {query_id_to_int[query_id]: query for query_id, query in query_id_to_query.items()}
    dataset = {"pmids": pmids,
               "query_id_to_qrels": query_id_to_qrels,
               "query_id_to_query": query_id_to_query}
    with open("TREC_COVID.pickle", "wb") as f:
        pickle.dump(dataset, f)


def _extract_pmids(documents: List[str]) -> List[int]:
    return [int(document.replace("http://www.ncbi.nlm.nih.gov/pubmed/", "")) for document in documents]


def parse_bioasq():
    with open(BIOASQ_SOURCE_FILE, "r", encoding="utf-8") as f:
        bioasq = json.load(f)
    questions = bioasq["questions"]
    query_id_to_query = {question["id"]: question["body"] for question in questions}
    query_id_to_pmids = {question["id"]: _extract_pmids(question["documents"]) for question in questions}

    # BioASQ is limited to 2024 baseline of PubMed, which is has highest file ID 1219. Find corresponding highest PMID:
    postgres_conn = psycopg.connect(host=os.getenv("POSTGRES_SERVER"),
                                    user=os.getenv("POSTGRES_USER"),
                                    password=os.getenv("POSTGRES_PASSWORD"),
                                    dbname=os.getenv("POSTGRES_DATABASE"))
    cursor = postgres_conn.cursor()
    cursor.execute(f"SET SEARCH_PATH={os.getenv('POSTGRES_SCHEMA')};")
    cursor.execute("SELECT MAX(pmid) FROM pubmed_articles WHERE file_number <= 1219")
    max_pmid = cursor.fetchone()[0]

    # Convert query IDs to ints:
    query_id_to_int = {}
    for i, query_id in enumerate(query_id_to_query):
        query_id_to_int[query_id] = i
    query_id_to_query = {query_id_to_int[query_id]: query for query_id, query in query_id_to_query.items()}
    query_id_to_pmids = {query_id_to_int[query_id]: pmids for query_id, pmids in query_id_to_pmids.items()}

    # Take a sample of query IDs:
    sampled_query_ids = random.sample(sorted(query_id_to_int.values()), 100)

    # Combine objects and save:
    dataset = {"query_id_to_query": query_id_to_query,
               "query_id_to_pmids": query_id_to_pmids,
               "max_pmid": max_pmid,
               "sampled_query_ids": sampled_query_ids}
    with open("BioASQTrain2024.pickle", "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    # parse_trec_covid()
    parse_bioasq()
