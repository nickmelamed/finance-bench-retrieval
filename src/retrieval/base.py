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

    def last_retrieval_usage(self) -> tuple[int, int]:
        """
        (prompt_tokens, completion_tokens) spent by the most
        recent `retrieve()` call. Static retrievers spend no
        LLM tokens, so the default is (0, 0); retrievers that
        call an LLM during retrieval (e.g. a tool-using agent)
        should override this.
        """
        return (0, 0)

    def last_retrieval_turn_count(self) -> int:
        """
        Number of LLM turns spent by the most recent `retrieve()`
        call. Static retrievers spend none, so the default is 0.
        """
        return 0