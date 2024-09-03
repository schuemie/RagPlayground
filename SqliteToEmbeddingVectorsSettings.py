from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SqliteToEmbeddingVectorsSettings:
    sqlite_path: str
    log_path: str
    parquet_folder: str
    batch_size: int
    embedding_model: str
    embed_document_prompt: Optional[str]
    embed_query_prompt: Optional[str]
    embedding_batch_size: int

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            return
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        processing = config["processing"]
        for key, value in processing.items():
            setattr(self, key, value)
        model = config["model"]
        for key, value in model.items():
            setattr(self, key, value)