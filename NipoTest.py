from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from transformers import AutoModel

DOCUMENT_FOLDER = "/Users/schuemie/git/benefit_risk_ai_pilot/Docs/Nipo source docs"


# Load the documents and split them into chunks.
loader = PyPDFDirectoryLoader(DOCUMENT_FOLDER)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4095,
    chunk_overlap=256
)
chunks = loader.load_and_split(text_splitter)

print(chunks)