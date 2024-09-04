from typing import List

import hnswlib
import numpy as np
import os


class HNSWIndex:
    """
    A class for managing an HNSWlib index. The class can create a new index or load an existing one
    from disk, and provides methods to add vectors in batches and search for nearest neighbors.

    Attributes:
    ------------
    index : hnswlib.Index
        The HNSWlib index object used for similarity search.
    dim : int
        The dimensionality of the vectors.
    max_elements : int
        The maximum number of elements that the index can hold.
    space : str
        The distance metric used for similarity search (e.g., 'cosine', 'l2').
    """

    def __init__(self, dim: int, max_elements: int, index_path: str = None, space: str = 'cosine'):
        """
        Initialize the HNSWIndex. Either creates a new index or loads an existing one from disk.

        Parameters:
        ------------
        dim : int
            The dimensionality of the vectors.
        max_elements : int
            The maximum number of elements that the index can hold.
        index_path : str, optional
            Path to the saved index. If provided, the index will be loaded from this path.
        space : str, optional
            The distance metric to use ('cosine' or 'l2'). Defaults to 'cosine'.
        """
        self.dim = dim
        self.max_elements = max_elements
        self.space = space

        self.index = hnswlib.Index(space=self.space, dim=self.dim)

        if index_path and os.path.exists(index_path):
            # Load the index from disk
            print(f"Loading index from {index_path}")
            self.index.load_index(index_path)
        else:
            # Initialize a new index
            print(f"Creating a new index with max_elements={max_elements}")
            self.index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)

    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray):
        """
        Add a batch of vectors to the index.

        Parameters:
        ------------
        vectors : np.ndarray
            A 2D NumPy array of vectors to add to the index.
        ids : np.ndarray
            A 1D NumPy array of IDs corresponding to the vectors.
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Each vector must have {self.dim} dimensions.")
        self.index.add_items(vectors, ids)

    def search(self, query_vector: np.ndarray | List[float], k: int = 5):
        """
        Search for the nearest neighbors of a query vector.

        Parameters:
        ------------
        query_vector : np.ndarray
            The query vector to search for nearest neighbors.
        k : int, optional
            The number of nearest neighbors to return. Defaults to 5.

        Returns:
        --------
        labels : np.ndarray
            The IDs of the nearest neighbors.
        distances : np.ndarray
            The distances to the nearest neighbors.
        """
        if isinstance(query_vector, np.ndarray):
            if query_vector.shape[0] != self.dim:
                raise ValueError(f"Query vector must have {self.dim} dimensions.")
        else:
            if len(query_vector) != self.dim:
                raise ValueError(f"Query vector must have {self.dim} dimensions.")
        def print_id(id):
            print(id)
            return True
        labels, distances = self.index.knn_query(query_vector, k=k, num_threads=1, filter=print_id())
        return labels, distances

    def save_index(self, path: str):
        """
        Save the index to disk.

        Parameters:
        ------------
        path : str
            The path where the index should be saved.
        """
        self.index.save_index(path)

    def set_ef(self, ef: int = 200):
        """
        Set the size of the dynamic list for search (controls accuracy/speed tradeoff).

        Parameters:
        ------------
        ef : int
            The size of the dynamic list for the search. Higher values lead to more accurate but slower searches.
        """
        self.index.set_ef(ef)

    def get_current_count(self):
        return self.index.get_current_count()