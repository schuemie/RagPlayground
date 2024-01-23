import os

import keyring
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from metapub import PubMedFetcher
from langchain.schema.document import Document
from transformers import AutoModel

QUERY = "malaria treatment"

COLLECTION_NAME = "rag_test"
DOCUMENT_FOLDER = "d:/RagDocs"

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=keyring.get_password("system", "LOCAL_POSTGRES_SERVER"),
    port=int(keyring.get_password("system", "LOCAL_POSTGRES_PORT")),
    database=keyring.get_password("system", "LOCAL_POSTGRES_DATABASE"),
    user=keyring.get_password("system", "LOCAL_POSTGRES_USER"),
    password=keyring.get_password("system", "LOCAL_POSTGRES_PASSWORD"),
)

fetch = PubMedFetcher()
pmids = fetch.pmids_for_query(QUERY, retmax=100)

docs = []
for pmid in pmids:
    file_name = f"{DOCUMENT_FOLDER}/{pmid}.txt"
    if os.path.exists(file_name):
        text = open(file_name).read()
        doc = Document(page_content=text,
                       metadata={"source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
        docs.append(doc)
    else:
        print(pmid)
        article = fetch.article_by_pmid(pmid)
        if article.abstract is None:
            text = article.title
        else:
            text = article.title + "\n" + article.abstract
        open(file_name, "w", encoding="UTF8").write(text)
        doc = Document(page_content=text,
                       metadata = {"source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
        docs.append(doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)
chunks = text_splitter.split_documents(docs)

# Workaround for https://github.com/langchain-ai/langchain/issues/6080
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
db = PGVector.from_documents(
    embedding=embeddings,
    documents=chunks,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)