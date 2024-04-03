from typing import Dict, Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModel
import os
import asyncio
from dataclasses import dataclass

model_path = "/Users/schuemie/git/LocalLlm2/models/zephyr-7b-beta.Q5_K_M.gguf"
llm = LlamaCpp(model_path=model_path,
               n_gpu_layers=-1,
               n_threads=8,
               n_ctx=4096,
               use_mlock=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever()
template = """<|system|>\n</s>\n</|user|>
{question}

Context: {context}
\n</s>\n<|assistant|>"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | llm | output_parser


class StoreLlmPromptHandler(BaseCallbackHandler):

    llm_prompt = ""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.llm_prompt = prompts[0]

    def get_llm_prompt(self):
        return self.llm_prompt


@dataclass
class LlmResponse:
    prompt: str
    response: str


async def get_llm_response(prompt: str, k: int) -> LlmResponse:
    handler = StoreLlmPromptHandler()
    retriever.search_kwargs = {"k": k}
    response = await chain.ainvoke(prompt, {"callbacks": [handler]})
    return LlmResponse(prompt=handler.get_llm_prompt(), response=response)


if __name__ == "__main__":
    response = asyncio.run(get_llm_response("What is the primary treatment option for multiple myeloma?"))
    print("Prompt:")
    print(response.prompt)
    print("\nResponse:")
    print(response.response)