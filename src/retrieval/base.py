from abc import ABC, abstractmethod

from src.types.schemas import RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract retrieval interface.

    All retrievers should:
    - accept a query
    - return RetrievalResult objects
    - expose consistent top_k behavior
    """

    def __init__(
        self,
        top_k: int = 5,
    ):
        self.top_k = top_k

    @abstractmethod
    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        raise NotImplementedError