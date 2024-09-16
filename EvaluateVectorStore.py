import os

import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from tqdm import tqdm

from TransformerEmbedder import TransformerEmbedder
from RetrievalEvaluation import RetrievalEvaluator, TrecCovidEvaluator, BioASQTrain2024Evaluator

load_dotenv()


def evaluate_vector_store(evaluator: RetrievalEvaluator, table_name: str, model_name: str):
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

    print(results)
