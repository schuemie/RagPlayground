import datetime
import sqlite3

# Some spot checks to see if PubMed data in SQLite looks good

connection = sqlite3.connect('e:/Medline/PubMed.sqlite')
cursor = connection.cursor()

def fetch_record(pmid: int):
    sql = f"""
    SELECT *
    FROM pubmed_articles
    WHERE pmid = {pmid};  	
    """
    cursor.execute(sql)
    record = cursor.fetchone()
    return record

record = fetch_record(38716090)

record[0] == 38716090

record[1] == "Massive Parallelization of Massive Sample-size Survival Analysis."

record[2] == "Large-scale observational health databases are increasingly popular for conducting comparative effectiveness and safety studies of medical products. However, increasing number of patients poses computational challenges when fitting survival regression models in such studies. In this paper, we use graphics processing units (GPUs) to parallelize the computational bottlenecks of massive sample-size survival analyses. Specifically, we develop and apply time- and memory-efficient single-pass parallel scan algorithms for Cox proportional hazards models and forward-backward parallel scan algorithms for Fine-Gray models for analysis with and without a competing risk using a cyclic coordinate descent optimization approach. We demonstrate that GPUs accelerate the computation of fitting these complex models in large databases by orders of magnitude as compared to traditional multi-core CPU parallelism. Our implementation enables efficient large-scale observational studies involving millions of patients and thousands of patient characteristics. The above implementation is available in the open-source R package Cyclops (Suchard et al., 2013)."

# No MeSH headings:
record[4] is None

record[5] == "Cox proportional hazards model\nFine-Gray model\nGraphics processing unit\nRegularized regression\nSurvival analysis"

# No chemicals:
record[6] is None

record[7] == "Yang, J\nSchuemie, MJ\nJi, X\nSuchard, MA"

record[8] == "Journal of computational and graphical statistics : a joint publication of American Statistical Association, Institute of Mathematical Statistics, Interface Foundation of North America"

record[9] == 2024

record[10] == "33"

record[11] == "1"

record[12] == "289-302"

record[13] == "Journal Article"

record[14] == 1402

sql = """
SELECT pmid,
    CASE WHEN title IS NULL THEN '' ELSE title || '\n\n' END ||
        CASE WHEN abstract IS NULL THEN '' ELSE abstract || '\n\n' END ||
        CASE WHEN mesh_terms IS NULL THEN '' ELSE 'MeSH terms:\n' || mesh_terms || '\n\n' END ||
        CASE WHEN keywords IS NULL THEN '' ELSE 'Keywords:\n' || keywords || '\n\n' END ||
        CASE WHEN chemicals IS NULL THEN '' ELSE 'Chemicals:\n' || chemicals || '\n\n' END AS text,
    publication_date
FROM pubmed_articles
WHERE pmid = 38716090;  	
"""
cursor.execute(sql)
record = cursor.fetchone()
print(record[1])