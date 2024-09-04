from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LoadVectorsInStoreSettings:
    parquet_folder: str
    vector_store_path: str
    log_path: str

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            return
        system = config["system"]
        for key, value in system.items():
            setattr(self, key, value)
