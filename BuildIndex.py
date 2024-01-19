import keyring
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from transformers import AutoModel

DOCUMENT_FOLDER = "/Users/schuemie/Data/RagTest"
COLLECTION_NAME = "rag_test"

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=keyring.get_password("system", "LOCAL_POSTGRES_SERVER"),
    port=int(keyring.get_password("system", "LOCAL_POSTGRES_PORT")),
    database=keyring.get_password("system", "LOCAL_POSTGRES_DATABASE"),
    user=keyring.get_password("system", "LOCAL_POSTGRES_USER"),
    password=keyring.get_password("system", "LOCAL_POSTGRES_PASSWORD"),
)

# Load the documents and split them into chunks.
loader = PyPDFDirectoryLoader(DOCUMENT_FOLDER)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256
)
chunks = loader.load_and_split(text_splitter)

# Create the embeddings and store them in the database.

# Workaround for https://github.com/langchain-ai/langchain/issues/6080
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
db = PGVector.from_documents(
    embedding=embeddings,
    documents=chunks,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)
        