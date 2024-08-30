from datetime import date

import chromadb

from TransformerEmbedder import TransformerEmbedder

DAYS_OFFSET = date(1970, 1, 1).toordinal()

def do_search() -> None:
    client = chromadb.PersistentClient(path="./PubMedChromaDb")
    collection = client.get_or_create_collection(name="pubmed_abstracts")

    embedder = TransformerEmbedder()

    query = "[Demonstration of tumor inhibiting properties of a strongly immunostimulating low-molecular weight substance. Comparative studies with ifosfamide on the immuno-labile DS carcinosarcoma. Stimulation of the autoimmune activity for approx. 20 days by BA 1, a N-(2-cyanoethylene)-urea. Novel prophylactic possibilities].\\nA report is given on the recent discovery of outstanding immunological properties in BA 1 [N-(2-cyanoethylene)-urea] having a (low) molecular mass M = 111.104. Experiments in 214 DS carcinosarcoma bearing Wistar rats have shown that BA 1, at a dosage of only about 12 percent LD50 (150 mg kg) and negligible lethality (1.7 percent), results in a recovery rate of 40 percent without hyperglycemia and, in one test, of 80 percent with hyperglycemia. Under otherwise unchanged conditions the reference substance ifosfamide (IF) -- a further development of cyclophosphamide -- applied without hyperglycemia in its most efficient dosage of 47 percent LD50 (150 mg kg) brought about a recovery rate of 25 percent at a lethality of 18 percent. (Contrary to BA 1, 250-min hyperglycemia caused no further improvement of the recovery rate.) However this comparison is characterized by the fact that both substances exhibit two quite different (complementary) mechanisms of action. Leucocyte counts made after application of the said cancerostatics and dosages have shown a pronounced stimulation with BA 1 and with ifosfamide, the known suppression in the post-therapeutic interval usually found with standard cancerostatics. In combination with the cited plaque test for BA 1, blood pictures then allow conclusions on the immunity status. Since IF can be taken as one of the most efficient cancerostatics--there is no other chemotherapeutic known up to now that has a more significant effect on the DS carcinosarcoma in rats -- these findings are of special importance. Finally, the total amount of leucocytes and lymphocytes as well as their time behaviour was determined from the blood picture of tumour-free rats after i.v. application of BA 1. The thus obtained numerical values clearly show that further research work on the prophylactic use of this substance seems to be necessary and very promising.\\nAnimals, Antineoplastic Agents, Carcinosarcoma, Cyclophosphamide, Drug Evaluation, Preclinical, Drug Tolerance, Erythrocyte Count, Hydrogen-Ion Concentration, Hyperglycemia, Ifosfamide, Immunity, Immunosuppression Therapy, Lethal Dose 50, Leukocyte Count, Male, Mice, Neoplasms, Experimental, Rats, Stimulation, Chemical, Time Factors, Urea'"
    query_embedding = embedder.embed_query(query)

    publication_date = date(1980, 1, 1)
    publication_date = publication_date.toordinal() - DAYS_OFFSET

    result = collection.query(query_embeddings=query_embedding,
                              n_results=10,
                              where={"publication_date": {"$lte": publication_date}})
    for i in range(10):
        print(f"ID = {result['ids'][0][i]}, distance = {str(result['distances'][0][i])}")
    # ID 22 should have the closest match


if __name__ == "__main__":
    do_search()
