from typing import List, Optional
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from numpy import ndarray
from sentence_transformers import SentenceTransformer


class TransformerEmbedder:
    """
        A class to generate embeddings for documents and queries using a pre-trained transformer model.

        The `TransformerEmbedder` class utilizes the `SentenceTransformer` model to encode texts (documents or queries) into vector embeddings. This is useful for various natural language processing tasks such as semantic search, clustering, and classification.

        Attributes:
        -----------
        model : SentenceTransformer
            The transformer model used for generating embeddings.
        embed_document_prompt : Optional[str]
            The prompt used by the model when embedding documents. If None, the model's default document embedding prompt is used.
        embed_query_prompt : Optional[str]
            The prompt used by the model when embedding queries. Defaults to "query".
        embedding_batch_size : int
            The batch size for embedding texts. Larger batch sizes may improve throughput but require more memory.

        Methods:
        --------
        embed_documents(texts: List[str]) -> ndarray:
            Generates embeddings for a list of documents.

            Parameters:
            -----------
            texts : List[str]
                A list of document strings to be embedded. If any element in the list is None, it will be replaced with an empty string before embedding.

            Returns:
            --------
            ndarray
                A NumPy array of embeddings corresponding to the input documents.

        embed_query(query: str) -> List[float]:
            Generates an embedding for a single query string.

            Parameters:
            -----------
            query : str
                The query string to be embedded.

            Returns:
            --------
            List[float]
                A list representing the embedding vector for the query.
        """
    def __init__(self,
                 model_name: str = "Snowflake/snowflake-arctic-embed-s",
                 embed_document_prompt: Optional[str] = None,
                 embed_query_prompt: Optional[str] = "query",
                 embedding_batch_size: int = 32):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.embed_document_prompt = embed_document_prompt
        self.embed_query_prompt = embed_query_prompt
        self.embedding_batch_size = embedding_batch_size

    def embed_documents(self, texts: List[str]) -> ndarray:
        texts = [text if text is not None else "" for text in texts]
        embeddings = self.model.encode(texts,
                                       batch_size=self.embedding_batch_size,
                                       prompt_name=self.embed_document_prompt)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(query, prompt_name=self.embed_query_prompt)
        return embedding.tolist()
