import pickle
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
from trectools import TrecEval, TrecQrel, TrecRun


def _predictions_dict_to_df(query_id_to_pmids: Dict[int, List[int]]) -> pd.DataFrame:
    predictions = {
        "query": [],
        "q0": [],
        "docid": [],
        "rank": [],
        "score": [],
        "system": [],
    }
    for query_id, pmids in query_id_to_pmids.items():
        n = len(pmids)
        predictions["query"].extend([query_id] * n)
        predictions["q0"].extend(["0"] * n)
        predictions["docid"].extend(pmids)
        predictions["rank"].extend(list(range(n)))
        predictions["score"].extend([0] * n)
        predictions["system"].extend(["system"] * n)
    return pd.DataFrame.from_dict(predictions)


def _trectools_object_to_dict(trec_eval: TrecEval) -> Dict[str, float]:
    result = {}
    result["num_ret"] = trec_eval.get_retrieved_documents(per_query=False)
    result["num_rel"] = trec_eval.get_relevant_documents(per_query=False)
    result["num_rel_ret"] = trec_eval.get_relevant_retrieved_documents(per_query=False)
    result["num_q"] = len(trec_eval.run.topics())
    result["map"] = trec_eval.get_map(depth=10000, per_query=False, trec_eval=True)
    result["gm_map"] = trec_eval.get_geometric_map(depth=10000, trec_eval=True)
    result["bpref"] = trec_eval.get_bpref(depth=1000, per_query=False, trec_eval=True)
    result["Rprec"] = trec_eval.get_rprec(depth=1000, per_query=False, trec_eval=True)
    result["recip_rank"] = trec_eval.get_reciprocal_rank(depth=1000, per_query=False, trec_eval=True)
    for v in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
        result[f"P@{v}"] = trec_eval.get_precision(depth=v, per_query=False, trec_eval=True)
    for v in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
        result[f"NDCG@{v}"] = trec_eval.get_ndcg(depth=v, per_query=False, trec_eval=True)
    return result


class RetrievalEvaluator(ABC):

    @abstractmethod
    def get_query_id_to_query(self) -> Dict[int, str]:
        """
        Returns the queries to use for the evaluation. Each query has a unique query ID (integer).

        :return: A dictionary from query ID to query text.
        """
        pass

    @abstractmethod
    def evaluate(self, query_id_to_pmids: Dict[int, List[int]]) -> Dict[str, float]:
        """
        Evaluate performance of the retrieval system, computing a large range of evaluation metrics.

        :param query_id_to_pmids: A dictionary from query ID to a ranked list of retrieved PMIDs.
        :return: A dictionary from metric name to metric value. See
        https://people.cs.georgetown.edu/~nazli/classes/ir-Slides/Evaluation-12.pdf for a nice overview of the various
        metrics.
        """
        pass


class TrecCovidEvaluator(RetrievalEvaluator):
    """
    Evaluation using subset of the TREC COVID set (https://ir.nist.gov/trec-covid/), limited to those documents that are
    in PubMed.

    During TREC, the best observed performance for an automatic run was:
    P@5: 0.7800
    NDCG@10: 0.6080
    MAP: 0.3128
    bpref: 0.4832

    But keep in mind this score was achieved before the set was publicly available. Future retrieval approaches will
    have been trained or fine-tuned on this set, and may show overly-optimistic results.
    """

    def __init__(self):
        with open("TREC_COVID.pickle", "rb") as f:
            dataset = pickle.load(f)
        self.query_id_to_qrels = dataset["query_id_to_qrels"]
        self.query_id_to_query = dataset["query_id_to_query"]
        self.allowed_pmids = set(dataset["pmids"])

    def get_query_id_to_query(self) -> Dict[int, str]:
        return self.query_id_to_query


    def _filter_pmids(self, query_id, pmids):
        return [pmid for pmid in pmids if pmid in self.allowed_pmids]

    def evaluate(self, query_id_to_pmids: Dict[int, List[int]]) -> Dict[str, float]:
        references = {
            "query": [],
            "q0": [],
            "docid": [],
            "rel": []
        }
        for query_id, pmid_to_score in self.query_id_to_qrels.items():
            n = len(pmid_to_score)
            references["query"].extend([query_id] * n)
            references["q0"].extend(["0"] * n)
            references["docid"].extend(pmid_to_score)
            references["rel"].extend(pmid_to_score.values())

        query_id_to_pmids = {query_id: self._filter_pmids(query_id, pmids) for query_id, pmids in query_id_to_pmids.items()}
        df_run = _predictions_dict_to_df(query_id_to_pmids)
        df_qrel = pd.DataFrame.from_dict(references)

        trec_run = TrecRun()
        trec_run.filename = "placeholder.file"
        trec_run.run_data = df_run

        trec_qrel = TrecQrel()
        trec_qrel.filename = "placeholder.file"
        trec_qrel.qrels_data = df_qrel

        trec_eval = TrecEval(trec_run, trec_qrel)
        return _trectools_object_to_dict(trec_eval)


class BioASQTrain2024Evaluator(RetrievalEvaluator):
    """
    Evaluation using the BioASQ Task B training set of 2024
    (http://participants-area.bioasq.org/general_information/Task12b/)
    """

    def __init__(self, use_sample: bool = False):
        with open("BioASQTrain2024.pickle", "rb") as f:
            dataset = pickle.load(f)
        self.query_id_to_query = dataset["query_id_to_query"]
        self.query_id_to_relevant_pmids = dataset["query_id_to_pmids"]
        if use_sample:
            sampled_query_ids = dataset["sampled_query_ids"]
            self.query_id_to_query = {query_id: self.query_id_to_query[query_id] for query_id in sampled_query_ids}
            self.query_id_to_relevant_pmids = {query_id: self.query_id_to_relevant_pmids[query_id] for query_id in sampled_query_ids}
        self.max_pmid = dataset["max_pmid"]

    def get_query_id_to_query(self) -> Dict[int, str]:
        return self.query_id_to_query

    def _filter_pmids(self, pmids: List[int]) -> List[int]:
        return [pmid for pmid in pmids if pmid <= self.max_pmid]

    def evaluate(self, query_id_to_pmids: Dict[int, List[int]]) -> Dict[str, float]:
        references = {
            "query": [],
            "q0": [],
            "docid": [],
            "rel": []
        }
        for query_id, relevant_pmids in self.query_id_to_relevant_pmids.items():
            pmids = query_id_to_pmids[query_id]
            non_relevant_pmids = [pmid for pmid in pmids if pmid not in relevant_pmids]
            n = len(relevant_pmids) + len(non_relevant_pmids)
            references["query"].extend([query_id] * n)
            references["q0"].extend(["0"] * n)
            references["docid"].extend(relevant_pmids)
            references["docid"].extend(non_relevant_pmids)
            references["rel"].extend([1]*len(relevant_pmids))
            references["rel"].extend([0] * len(non_relevant_pmids))

        df_qrel = pd.DataFrame.from_dict(references)
        query_id_to_pmids = {query_id: self._filter_pmids(pmids) for query_id, pmids in query_id_to_pmids.items()}
        df_run = _predictions_dict_to_df(query_id_to_pmids)

        trec_run = TrecRun()
        trec_run.filename = "placeholder.file"
        trec_run.run_data = df_run

        trec_qrel = TrecQrel()
        trec_qrel.filename = "placeholder.file"
        trec_qrel.qrels_data = df_qrel

        trec_eval = TrecEval(trec_run, trec_qrel)
        return _trectools_object_to_dict(trec_eval)


if __name__ == "__main__":
    # with open("TREC_COVID.pickle", "rb") as f:
    #     dataset = pickle.load(f)
    # query_id_to_qrels = dataset["query_id_to_qrels"]
    # query_id_to_pmids = {query_id: pmid_to_score.keys() for query_id, pmid_to_score in query_id_to_qrels.items()}
    # evaluator = TrecCovidEvaluator()
    # results = evaluator.evaluate(query_id_to_pmids)
    # print(results)

    with open("BioASQTrain2024.pickle", "rb") as f:
        dataset = pickle.load(f)
    query_id_to_pmids = dataset["query_id_to_pmids"]
    evaluator = BioASQTrain2024Evaluator()
    results = evaluator.evaluate(query_id_to_pmids)
    print(results)