import keyring
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from transformers import AutoModel


COLLECTION_NAME = "rag_test"

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=keyring.get_password("system", "LOCAL_POSTGRES_SERVER"),
    port=int(keyring.get_password("system", "LOCAL_POSTGRES_PORT")),
    database=keyring.get_password("system", "LOCAL_POSTGRES_DATABASE"),
    user=keyring.get_password("system", "LOCAL_POSTGRES_USER"),
    password=keyring.get_password("system", "LOCAL_POSTGRES_PASSWORD"),
)

# Workaround for https://github.com/langchain-ai/langchain/issues/6080
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
db = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# query = "What is the recommended therapy for relapsed multiple myeloma?"
query = "Which mosquito carries malaria?"
# query = "The study analyzed a sample of 171 children under 5 years old to assess various characteristics and variables related to pre-referral treatment. The findings reveal notable proportions in gender distribution, age categories, RDT results, presence of diarrhea, fast breathing, fever, danger signs, and timely medical visits. The results highlight the need to strengthen pre-referral treatment interventions and enhance iCCM programs."
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
