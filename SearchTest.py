import os

import psycopg
from pgvector.psycopg import register_vector

from sentence_transformers import SentenceTransformer
from TransformerEmbedder import TransformerEmbedder
from dotenv import load_dotenv


load_dotenv()

def do_search() -> None:

    conn = psycopg.connect(host=os.getenv("POSTGRES_SERVER"),
                           user=os.getenv("POSTGRES_USER"),
                           password=os.getenv("POSTGRES_PASSWORD"),
                           dbname=os.getenv("POSTGRES_DATABASE"))
    register_vector(conn)

    embedder = TransformerEmbedder()

    # query = "[Demonstration of tumor inhibiting properties of a strongly immunostimulating low-molecular weight substance. Comparative studies with ifosfamide on the immuno-labile DS carcinosarcoma. Stimulation of the autoimmune activity for approx. 20 days by BA 1, a N-(2-cyanoethylene)-urea. Novel prophylactic possibilities].\\nA report is given on the recent discovery of outstanding immunological properties in BA 1 [N-(2-cyanoethylene)-urea] having a (low) molecular mass M = 111.104. Experiments in 214 DS carcinosarcoma bearing Wistar rats have shown that BA 1, at a dosage of only about 12 percent LD50 (150 mg kg) and negligible lethality (1.7 percent), results in a recovery rate of 40 percent without hyperglycemia and, in one test, of 80 percent with hyperglycemia. Under otherwise unchanged conditions the reference substance ifosfamide (IF) -- a further development of cyclophosphamide -- applied without hyperglycemia in its most efficient dosage of 47 percent LD50 (150 mg kg) brought about a recovery rate of 25 percent at a lethality of 18 percent. (Contrary to BA 1, 250-min hyperglycemia caused no further improvement of the recovery rate.) However this comparison is characterized by the fact that both substances exhibit two quite different (complementary) mechanisms of action. Leucocyte counts made after application of the said cancerostatics and dosages have shown a pronounced stimulation with BA 1 and with ifosfamide, the known suppression in the post-therapeutic interval usually found with standard cancerostatics. In combination with the cited plaque test for BA 1, blood pictures then allow conclusions on the immunity status. Since IF can be taken as one of the most efficient cancerostatics--there is no other chemotherapeutic known up to now that has a more significant effect on the DS carcinosarcoma in rats -- these findings are of special importance. Finally, the total amount of leucocytes and lymphocytes as well as their time behaviour was determined from the blood picture of tumour-free rats after i.v. application of BA 1. The thus obtained numerical values clearly show that further research work on the prophylactic use of this substance seems to be necessary and very promising.\\nAnimals, Antineoplastic Agents, Carcinosarcoma, Cyclophosphamide, Drug Evaluation, Preclinical, Drug Tolerance, Erythrocyte Count, Hydrogen-Ion Concentration, Hyperglycemia, Ifosfamide, Immunity, Immunosuppression Therapy, Lethal Dose 50, Leukocyte Count, Male, Mice, Neoplasms, Experimental, Rats, Stimulation, Chemical, Time Factors, Urea'"
    # query = "Does ibuprofen cause gastrointestinal bleeding?"
    query = "Serially Combining Epidemiological Designs Does Not Improve Overall Signal Detection in Vaccine Safety Surveillance"
    query_embedding = embedder.embed_query(query)

    query = """
            SELECT pmid, embedding <=> %s AS similarity
            FROM pubmed.vectors_snowflake_arctic_s
            ORDER BY embedding <=> %s
            LIMIT 5;
            """
    embedding_str = f"[{','.join(map(str, query_embedding))}]"
    result = conn.execute(query, (embedding_str, embedding_str))

    # Fetch and print the top similar results
    similar_rows = result.fetchall()

    for row in similar_rows:
        pmid, similarity = row
        print(f"PMID: {pmid}, Similarity: {similarity}")

    # ID 22 should have the closest match

def explore_short_abstracts():
    query = "What is the comparative risk of gastrointestinal bleeding associated with the use of dabigatran, rivaroxaban, and warfarin?"
    text = """
    New oral anticoagulants increase risk for gastrointestinal bleeding: a systematic review and meta-analysis
    Background & aims: A new generation of oral anticoagulants (nOAC), which includes thrombin and factor Xa inhibitors, has been shown to be effective, but little is known about whether these drugs increase patients' risk for gastrointestinal bleeding (GIB). Patients who require OAC therapy frequently have significant comorbidities and may also take aspirin and/or thienopyridines. We performed a systematic review and meta-analysis of the risk of GIB and clinically relevant bleeding in patients taking nOAC.

    Methods: We queried MEDLINE, EMbase, and the Cochrane library (through July 2012) without language restrictions. We analyzed data from 43 randomized controlled trials (151,578 patients) that compared nOAC (regardless of indication) with standard care for risk of bleeding (19 trials on GIB). Odds ratios (ORs) were estimated using a random-effects model. Heterogeneity was assessed with the Cochran Q test and the Higgins I(2) test.

    Results: The overall OR for GIB among patients taking nOAC was 1.45 (95% confidence interval [CI], 1.07-1.97), but there was substantial heterogeneity among studies (I2, 61%). Subgroup analyses showed that the OR for atrial fibrillation was 1.21 (95% CI, 0.91-1.61), for thromboprophylaxis after orthopedic surgery the OR was 0.78 (95% CI, 0.31-1.96), for treatment of venous thrombosis the OR was 1.59 (95% CI, 1.03-2.44), and for acute coronary syndrome the OR was 5.21 (95% CI, 2.58-10.53). Among the drugs studied, the OR for apixaban was 1.23 (95% CI, 0.56-2.73), the OR for dabigatran was 1.58 (95% CI, 1.29-1.93), the OR for edoxaban was 0.31 (95% CI, 0.01-7.69), and the OR for rivaroxaban was 1.48 (95% CI, 1.21-1.82). The overall OR for clinically relevant bleeding in patients taking nOAC was 1.16 (95% CI, 1.00-1.34), with similar trends among subgroups.

    Conclusions: Studies on treatment of venous thrombosis or acute coronary syndrome have shown that patients treated with nOAC have an increased risk of GIB, compared with those who receive standard care. Better reporting of GIB events in future trials could allow stratification of patients for therapy with gastroprotective agents.
    """
    text2 = """
    New oral anticoagulants increase risk for gastrointestinal bleeding: a systematic review and meta-analysis
    Methods: We queried MEDLINE, EMbase, and the Cochrane library (through July 2012) without language restrictions. We analyzed data from 43 randomized controlled trials (151,578 patients) that compared nOAC (regardless of indication) with standard care for risk of bleeding (19 trials on GIB). Odds ratios (ORs) were estimated using a random-effects model. Heterogeneity was assessed with the Cochran Q test and the Higgins I(2) test.

    Results: The overall OR for GIB among patients taking nOAC was 1.45 (95% confidence interval [CI], 1.07-1.97), but there was substantial heterogeneity among studies (I2, 61%). Subgroup analyses showed that the OR for atrial fibrillation was 1.21 (95% CI, 0.91-1.61), for thromboprophylaxis after orthopedic surgery the OR was 0.78 (95% CI, 0.31-1.96), for treatment of venous thrombosis the OR was 1.59 (95% CI, 1.03-2.44), and for acute coronary syndrome the OR was 5.21 (95% CI, 2.58-10.53). Among the drugs studied, the OR for apixaban was 1.23 (95% CI, 0.56-2.73), the OR for dabigatran was 1.58 (95% CI, 1.29-1.93), the OR for edoxaban was 0.31 (95% CI, 0.01-7.69), and the OR for rivaroxaban was 1.48 (95% CI, 1.21-1.82). The overall OR for clinically relevant bleeding in patients taking nOAC was 1.16 (95% CI, 1.00-1.34), with similar trends among subgroups.

    Conclusions: Studies on treatment of venous thrombosis or acute coronary syndrome have shown that patients treated with nOAC have an increased risk of GIB, compared with those who receive standard care. Better reporting of GIB events in future trials could allow stratification of patients for therapy with gastroprotective agents.
    """
    text3 = """
    New oral anticoagulants increase risk for gastrointestinal bleeding: a systematic review and meta-analysis
    Results: The overall OR for GIB among patients taking nOAC was 1.45 (95% confidence interval [CI], 1.07-1.97), but there was substantial heterogeneity among studies (I2, 61%). Subgroup analyses showed that the OR for atrial fibrillation was 1.21 (95% CI, 0.91-1.61), for thromboprophylaxis after orthopedic surgery the OR was 0.78 (95% CI, 0.31-1.96), for treatment of venous thrombosis the OR was 1.59 (95% CI, 1.03-2.44), and for acute coronary syndrome the OR was 5.21 (95% CI, 2.58-10.53). Among the drugs studied, the OR for apixaban was 1.23 (95% CI, 0.56-2.73), the OR for dabigatran was 1.58 (95% CI, 1.29-1.93), the OR for edoxaban was 0.31 (95% CI, 0.01-7.69), and the OR for rivaroxaban was 1.48 (95% CI, 1.21-1.82). The overall OR for clinically relevant bleeding in patients taking nOAC was 1.16 (95% CI, 1.00-1.34), with similar trends among subgroups.

    Conclusions: Studies on treatment of venous thrombosis or acute coronary syndrome have shown that patients treated with nOAC have an increased risk of GIB, compared with those who receive standard care. Better reporting of GIB events in future trials could allow stratification of patients for therapy with gastroprotective agents.
    """
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s", trust_remote_code=True)
    e1 = model.encode(query, prompt_name="query")
    print(model.similarity(e1, e1))
    e2 = model.encode(text)
    print(model.similarity(e1, e2))
    e2 = model.encode(text2)
    print(model.similarity(e1, e2))
    e2 = model.encode(text3)
    print(model.similarity(e1, e2))

if __name__ == "__main__":
    # do_search()
    explore_short_abstracts()