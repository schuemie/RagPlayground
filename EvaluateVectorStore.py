import json
import os
import urllib.parse
from datetime import datetime
from time import sleep
from typing import Optional, Dict
from xml.etree import ElementTree

import psycopg
import requests
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from tqdm import tqdm

from TransformerEmbedder import TransformerEmbedder
from RetrievalEvaluation import RetrievalEvaluator, TrecCovidEvaluator, BioASQTrain2024Evaluator

load_dotenv()


def evaluate_vector_store(evaluator: RetrievalEvaluator, table_name: str, model_name: str) -> Dict[str, float]:
    conn = psycopg.connect(host=os.getenv("POSTGRES_SERVER"),
                           user=os.getenv("POSTGRES_USER"),
                           password=os.getenv("POSTGRES_PASSWORD"),
                           dbname=os.getenv("POSTGRES_DATABASE"))
    register_vector(conn)
    conn.execute("SET hnsw.ef_search = 1000")

    embedder = TransformerEmbedder(model_name=model_name)

    query_id_to_query = evaluator.get_query_id_to_query()
    query_id_to_pmids = {}
    for query_id, query in tqdm(query_id_to_query.items()):
        query_embedding = embedder.embed_query(query)
        sql = f"""
                SELECT pmid
                FROM pubmed.{table_name}
                ORDER BY embedding <=> %s
                LIMIT 1000;
                """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        result = conn.execute(sql, (embedding_str, ))
        similar_rows = result.fetchall()
        pmids = [row[0] for row in similar_rows]
        query_id_to_pmids[query_id] = pmids
    return evaluator.evaluate(query_id_to_pmids)

def _get_gpt4_response(prompt, system_prompt=None):
    # Construct the messages for the API request
    if system_prompt is None:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""}
        ]

    # Prepare the JSON payload for the request
    payload = {
        "messages": messages,
        "temperature": 0.00000001,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    # Get the API key and endpoint from environment variables
    api_endpoint = os.environ.get("GENAI_GPT4_ENDPOINT")

    headers = {
        'api-key': os.environ.get("GENAI_GPT4_KEY"),
        'Content-Type': 'application/json'
    }

    # Send the request to the GPT-4 API
    response = requests.request("POST", url=api_endpoint, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"error: {response.status_code}, details: {response.text}")

def _search_pubmed(query, return_max=10, sort="relevance"):
    # Prepare the PubMed API URL
    url_template = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax={}&sort={}&term={}"
    encoded_query = urllib.parse.quote(query)
    url = url_template.format(return_max, sort, encoded_query)

    # Send the GET request to PubMed
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the XML response
        root = ElementTree.fromstring(response.content)
        id_list = root.find('.//IdList')

        if id_list is not None:
            return [int(pmid.text) for pmid in id_list.findall('.//Id')]
        else:
            return []

def evaluate_llm_pubmed_queries(evaluator: RetrievalEvaluator,
                                cache_folder: str,
                                prompt_template: Optional[str] = None,
                                system_prompt: Optional[str] = None) -> Dict[str, float]:
    os.makedirs(cache_folder, exist_ok=True)

    if prompt_template is None:
        prompt_template = """
        Write a PubMed search query that retrieves literature relevant to the research question below. Avoid using overly generic terms or MeSH terms. Use ‘OR’ operators to cover relevant synonyms and variations, and minimize the use of restrictive ‘AND’ clauses. Return only the query, so I can send it directly to PubMed.

        Research question: %s

        Pubmed query:
        """

    if system_prompt is None:
        system_prompt = """
        You are an expert assistant in scientific writing and literature search, specifically for PubMed queries. Your task is to generate search queries to retrieve relevant literature from PubMed based for a given research question.

        Guidelines:

            1.	Do not use MeSH terms. Instead, focus on natural language keywords and key concepts that are central to the argument.
            2.	Avoid overly generic terms or phrases. Ensure the terms are specific to the argument, while still using ‘OR’ combinations to account for relevant synonyms and variations.
            3.	While creating specific queries, minimize the use of restrictive ‘AND’ operators. Focus on creating balanced, concept-driven queries that remain broad enough to capture relevant literature but without becoming too general.

        The goal is to generate focused, specific PubMed search queries that retrieve relevant literature without being too restrictive or too generic.
        """

    last_query_time = datetime.now()
    query_id_to_query = evaluator.get_query_id_to_query()
    query_id_to_pmids = {}
    for query_id, query in tqdm(query_id_to_query.items()):
        file_name = os.path.join(cache_folder, f"PubMedQuery_query{query_id}.txt")
        if os.path.isfile(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                pubmed_query = f.read()
        else:
            prompt = prompt_template % query
            pubmed_query = _get_gpt4_response(prompt, system_prompt)
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(pubmed_query)

        file_name = os.path.join(cache_folder, f"PubMedResults_query{query_id}.txt")
        if os.path.isfile(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                pmids = [int(pmid) for pmid in f.read().split("\n")]
        else:
            if (datetime.now() - last_query_time).total_seconds() < 1:
                sleep(1)
            pmids = _search_pubmed(pubmed_query, return_max=1000)
            last_query_time = datetime.now()
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("\n".join([str(pmid) for pmid in pmids]))

        query_id_to_pmids[query_id] = pmids
    return evaluator.evaluate(query_id_to_pmids)

if __name__ == "__main__":

    """
    Evaluating using TREC_COVID
    
    During TREC, the best observed performance for an automatic run was:
    P@5: 0.7800
    NDCG@10: 0.6080
    MAP: 0.3128
    bpref: 0.4832
    """

    # results = evaluate_vector_store(TrecCovidEvaluator(),
    #                                 table_name="vectors_snowflake_arctic_s",
    #                                 model_name="Snowflake/snowflake-arctic-embed-s")
    # {'num_ret': 5440, 'num_rel': 11482, 'num_rel_ret': 2395, 'num_q': 50, 'map': 0.12177238916138107,
    #  'gm_map': 0.08645839514954443, 'bpref': 0.2124913169702809, 'Rprec': 0.2132773633805483,
    #  'recip_rank': 0.6777303807303806, 'P@5': 0.4800000000000001, 'P@10': 0.456, 'P@15': 0.4573333333333334,
    #  'P@20': 0.441, 'P@30': 0.43466666666666676, 'P@100': 0.37300000000000005, 'P@200': 0.23379999999999998,
    #  'P@500': 0.09580000000000002, 'P@1000': 0.04790000000000001, 'NDCG@5': 0.7893890126964439,
    #  'NDCG@10': 0.7306101159804396, 'NDCG@15': 0.6962690609787114, 'NDCG@20': 0.6579100122862103,
    #  'NDCG@30': 0.6113685790488763, 'NDCG@100': 0.4481719256441532, 'NDCG@200': 0.3573552151562602,
    #  'NDCG@500': 0.32318486602318436, 'NDCG@1000': 0.32262728679628155}

    # results = evaluate_vector_store(TrecCovidEvaluator(),
    #                                 table_name="vectors_snowflake_arctic_m",
    #                                 model_name="Snowflake/snowflake-arctic-embed-m-v1.5")
    # {'num_ret': 6446, 'num_rel': 11482, 'num_rel_ret': 2789, 'num_q': 50, 'map': 0.13935966083355408,
    #  'gm_map': 0.10144905986956507, 'bpref': 0.2492830200344899, 'Rprec': 0.24791142864380963,
    #  'recip_rank': 0.6453594470046083, 'P@5': 0.48, 'P@10': 0.466, 'P@15': 0.4493333333333333,
    #  'P@20': 0.44400000000000006, 'P@30': 0.4453333333333333, 'P@100': 0.3913999999999999, 'P@200': 0.2657,
    #  'P@500': 0.11155999999999996, 'P@1000': 0.05577999999999998, 'NDCG@5': 0.7697028300266837,
    #  'NDCG@10': 0.7351341940263275, 'NDCG@15': 0.7117385089178063, 'NDCG@20': 0.6841963025113302,
    #  'NDCG@30': 0.6389619597101399, 'NDCG@100': 0.4786779034498604, 'NDCG@200': 0.39673378054478114,
    #  'NDCG@500': 0.3625217339643562, 'NDCG@1000': 0.3617809747524116}

    # results = evaluate_llm_pubmed_queries(TrecCovidEvaluator(), "e:/temp/retrievalevalcache")
    # {'num_ret': 3045, 'num_rel': 11482, 'num_rel_ret': 939, 'num_q': 47, 'map': 0.0408806802726849, 'gm_map': nan,
    #  'bpref': 0.08420444884099436, 'Rprec': 0.08530745214318296, 'recip_rank': 0.4980882663874562,
    #  'P@5': 0.2936170212765957, 'P@10': 0.2872340425531915, 'P@15': 0.2780141843971631, 'P@20': 0.2659574468085107,
    #  'P@30': 0.24397163120567372, 'P@100': 0.14659574468085107, 'P@200': 0.09361702127659573,
    #  'P@500': 0.03995744680851064, 'P@1000': 0.01997872340425532, 'NDCG@5': 0.41606080906221493,
    #  'NDCG@10': 0.3763471029281088, 'NDCG@15': 0.3611901869661997, 'NDCG@20': 0.34248454606210116,
    #  'NDCG@30': 0.31066998926732414, 'NDCG@100': 0.19609328905105683, 'NDCG@200': 0.1526895342304493,
    #  'NDCG@500': 0.13710663032775505, 'NDCG@1000': 0.13680085599138453}

    """
    Evaluating using BioASQ 2024 task B training set

    Currently using a sample of 100 topics
    """

    # results = evaluate_vector_store(BioASQTrain2024Evaluator(use_sample=True),
    #                                 table_name="vectors_snowflake_arctic_s",
    #                                 model_name="Snowflake/snowflake-arctic-embed-s")
    # {'num_ret': 96579, 'num_rel': 864, 'num_rel_ret': 592, 'num_q': 100, 'map': 0.006790181961261331,
    #  'gm_map': 0.0021980402710127774, 'bpref': 0.0, 'Rprec': 0.0, 'recip_rank': 0.004131608443273454, 'P@5': 0.0,
    #  'P@10': 0.0, 'P@15': 0.0, 'P@20': 0.0, 'P@30': 0.0, 'P@100': 0.0013, 'P@200': 0.0029500000000000004,
    #  'P@500': 0.006200000000000001, 'P@1000': 0.005920000000000001, 'NDCG@5': 0.22251067379019138,
    #  'NDCG@10': 0.21830093498726455, 'NDCG@15': 0.22702146389363942, 'NDCG@20': 0.23406657184218155,
    #  'NDCG@30': 0.25283688586618003, 'NDCG@100': 0.2948007080392426, 'NDCG@200': 0.3169742164036957,
    #  'NDCG@500': 0.3383820470790254, 'NDCG@1000': 0.34792121309494106}

    # results = evaluate_vector_store(BioASQTrain2024Evaluator(use_sample=True),
    #                                 table_name="vectors_snowflake_arctic_m",
    #                                 model_name="Snowflake/snowflake-arctic-embed-m-v1.5")
    # {'num_ret': 97104, 'num_rel': 864, 'num_rel_ret': 618, 'num_q': 100, 'map': 0.007182810620262983,
    #  'gm_map': 0.0020942166631073627, 'bpref': 0.0, 'Rprec': 0.0, 'recip_rank': 0.0041412807059642755, 'P@5': 0.0,
    #  'P@10': 0.0, 'P@15': 0.0, 'P@20': 0.0, 'P@30': 0.0, 'P@100': 0.0012000000000000001, 'P@200': 0.00335,
    #  'P@500': 0.006740000000000001, 'P@1000': 0.0061800000000000015, 'NDCG@5': 0.286697425727848,
    #  'NDCG@10': 0.2822955808328984, 'NDCG@15': 0.284750008582755, 'NDCG@20': 0.2943464234251836,
    #  'NDCG@30': 0.3086030265566418, 'NDCG@100': 0.34954064096926074, 'NDCG@200': 0.37217871567165545,
    #  'NDCG@500': 0.3943900085240318, 'NDCG@1000': 0.40202841078889046}

    # results = evaluate_llm_pubmed_queries(BioASQTrain2024Evaluator(use_sample=True), "e:/temp/retrievalevalcache_bioasq")
    # {'num_ret': 30718, 'num_rel': 864, 'num_rel_ret': 304, 'num_q': 88, 'map': 0.07484124848343826, 'gm_map': nan,
    #  'bpref': 0.01964275139502872, 'Rprec': 0.07437739808702727, 'recip_rank': 0.11660581082663732,
    #  'P@5': 0.05454545454545456, 'P@10': 0.04318181818181818, 'P@15': 0.031060606060606063,
    #  'P@20': 0.027840909090909093, 'P@30': 0.025, 'P@100': 0.016022727272727272, 'P@200': 0.00931818181818182,
    #  'P@500': 0.004318181818181818, 'P@1000': 0.003454545454545455, 'NDCG@5': 0.16004032046355862,
    #  'NDCG@10': 0.16485711854817128, 'NDCG@15': 0.16478692570908746, 'NDCG@20': 0.17220611472909364,
    #  'NDCG@30': 0.18506909262602303, 'NDCG@100': 0.2209366724606398, 'NDCG@200': 0.23385335285310632,
    #  'NDCG@500': 0.2450003307253397, 'NDCG@1000': 0.24965773461719654}

    print(results)
