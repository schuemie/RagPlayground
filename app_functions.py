from typing import Dict, Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModel
import os
import asyncio
from dataclasses import dataclass

if os.environ.get("genai_gpt4_endpoint", "default") == "default":
    # For local testing: use LlamaCpp running Zephyr and Jina embeddings on the local machine.
    use_gpt4 = False
    model_path = "/Users/schuemie/git/LocalLlm2/models/zephyr-7b-beta.Q5_K_M.gguf"
    llm = LlamaCpp(model_path=model_path,
                   n_gpu_layers=-1,
                   n_threads=8,
                   n_ctx=4096,
                   use_mlock=True)
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-base-en")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    template = """<|system|>\n</s>\n</|user|>user
    Answer the question based only on the following context:
    {context}

    Provide the source(s) of your answer.

    Question: {question}\n</s>\n<|assistant|>"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    chain = setup_and_retrieval | prompt | llm | output_parser
else:
    # For production: use Azure OpenAI with GPT-4 and Azure embeddings.
    use_gpt4 = True
    llm = AzureChatOpenAI(
        openai_api_key=os.environ.get("genai_api_key"),
        api_version="2023-03-15-preview",
        azure_endpoint=os.environ.get("genai_gpt4_endpoint"),
    )


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


async def get_llm_response(prompt: str) -> LlmResponse:
    if use_gpt4:
        message = HumanMessage(
            content=prompt
        )
        response = await llm.ainvoke([message])
        return response.content
    else:
        # message = "<|system|>You are a helpful assistant.</s><|user|>" + prompt + "</s><|assistant|>"
        # response = await llm.ainvoke(message)
        # return response
        handler = StoreLlmPromptHandler()
        response = await chain.ainvoke(prompt, {"callbacks": [handler]})
        return LlmResponse(prompt=handler.get_llm_prompt(), response=response)


if __name__ == "__main__":
    response = asyncio.run(get_llm_response("What is the primary treatment option for multiple myeloma?"))
    print("Prompt:")
    print(response.prompt)
    print("\nResponse:")
    print(response.response)