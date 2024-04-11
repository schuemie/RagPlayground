from typing import Dict, Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModel

model_path = "/Users/schuemie/git/LocalLlm2/models/zephyr-7b-beta.Q5_K_M.gguf"
llm = LlamaCpp(model_path=model_path,
               n_gpu_layers=-1,
               n_threads=8,
               n_ctx=4096,
               use_mlock=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
template = """<|system|>\n</s>\n</|user|>
{prompt}

Context: {context}
\n</s>\n<|assistant|>"""


async def get_llm_response(prompt: str, search_results: list[tuple[Document, float]]) -> str:
    prompt_with_context = template.format(prompt=prompt, context=str(search_results))
    result = await llm.ainvoke(prompt_with_context)
    return result


def get_search_results(prompt: str, k: int) -> list[tuple[Document, float]]:
    docs_with_score = db.similarity_search_with_score(prompt, k=k)
    return docs_with_score


if __name__ == "__main__":
    # response = asyncio.run(get_llm_response("What is the primary treatment option for multiple myeloma?"))
    # print("Prompt:")
    # print(response.prompt)
    # print("\nResponse:")
    # print(response.response)
    response = get_search_results("What is the primary treatment option for multiple myeloma?", 5)
    for doc, score in response:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)



