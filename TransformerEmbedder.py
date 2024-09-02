from typing import List

from numpy import ndarray
from sentence_transformers import SentenceTransformer


class TransformerEmbedder:
    def __init__(self,
                 model_name: str = "Snowflake/snowflake-arctic-embed-s",
                 batch_size: int = 32):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> ndarray:
        texts = [text if text is not None else "" for text in texts]
        embeddings = self.model.encode(texts, batch_size=self.batch_size)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        query = "search_query: " + query
        embedding = self.model.encode(query, prompt_name="query")
        return embedding.tolist()
