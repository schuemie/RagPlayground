import keyring
from langchain_openai.llms import AzureOpenAI

llm = AzureOpenAI(
    openai_api_key=keyring.get_password("system", "genai_api_key"),
    api_version="2023-03-15-preview",
    azure_endpoint=keyring.get_password("system", "genai_azure_endpoint"),
)
print(llm.invoke("Tell me about malaria treatment."))