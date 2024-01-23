import keyring
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.llms import AzureOpenAI
from transformers import AutoModel
from langchain.globals import set_debug

set_debug(True)

# Connect to vectorstore and embedder
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
vectorstore = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

template = """<|im_start|>user
Answer the question based only on the following context:
{context}

Provide the source(s) of your answer.

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
prompt = ChatPromptTemplate.from_template(template)

model = AzureOpenAI(
    openai_api_key=keyring.get_password("system", "genai_api_key"),
    api_version="2023-03-15-preview",
    azure_endpoint=keyring.get_password("system", "genai_azure_endpoint"),
)

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | model | output_parser

#print(chain.invoke("Which mosquito carries malaria?"))
print(chain.invoke("How is malaria diagnosed?"))
