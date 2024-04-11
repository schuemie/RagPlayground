from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModel

from CustomPDFDirectoryLoader import CustomPDFDirectoryLoader

DOCUMENT_FOLDER = "/Users/schuemie/Data/RagTest"

# Load the documents and split them into chunks.
# loader = PyPDFDirectoryLoader(DOCUMENT_FOLDER)
loader = CustomPDFDirectoryLoader(DOCUMENT_FOLDER, extract_images=True)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256
)
chunks = loader.load_and_split(text_splitter)
print("Number of chunks:", len(chunks))

# Create the embeddings and store them in the database.

# Workaround for https://github.com/langchain-ai/langchain/issues/6080
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
# embedding_function = embeddings.get_embedding_function()

db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
db.persist()
