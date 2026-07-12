from finance_bench.types.schemas import TokenUsage


class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.retrieval_tokens = 0

    def add_prompt_tokens(self, n: int):
        self.prompt_tokens += n

    def add_completion_tokens(self, n: int):
        self.completion_tokens += n

    def add_retrieval_tokens(self, n: int):
        self.retrieval_tokens += n

    @property
    def total_tokens(self):
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.retrieval_tokens
        )

    def summary(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            retrieval_tokens=self.retrieval_tokens,
            total_tokens=self.total_tokens,
        )