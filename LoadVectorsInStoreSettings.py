from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LoadVectorsInStoreSettings:
    parquet_folder: str
    dimensions: int
    store_type: str
    path: Optional[str]
    path: Optional[str]
    schema: Optional[str]
    table: Optional[str]
    log_path: str

    PGVECTOR = "pgvector"
    HSNWLIB = "hsnwlib"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            return
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
        vector_store = config["vector_store"]
        for key, value in vector_store.items():
            setattr(self, key, value)

    def __post_init__(self):
        if self.store_type not in [self.PGVECTOR, self.HSNWLIB]:
            raise ValueError(f"vector_store.type must be '{self.PGVECTOR}' or '{self.HSNWLIB}'")
        if self.store_type == self.PGVECTOR and (
                self.schema is None or self.table is None):
            raise ValueError("vector_store.schema and vector_store.table must be specified when "
                             f"vector_store.type = '{self.PGVECTOR}'")
        if self.store_type == self.HSNWLIB and self.path is None:
            raise ValueError(f"vector_store.path must be specified when vector_store.type = '{self.HSNWLIB}'")

