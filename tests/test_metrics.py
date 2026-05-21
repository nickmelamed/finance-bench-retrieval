from src.evaluation.bootstrap import BootstrapCI

from evaluation.qa_metrics import (
    accuracy,
    token_efficiency,
)

from src.llm.token_tracking import (
    TokenTracker,
)

# bootstrap validation

values = [
    1,
    1,
    0,
    1,
    1,
    0,
    1,
]

bootstrap = BootstrapCI()

result = bootstrap.compute(values)


# Accuracy validation 

accuracy(values)

# Token accounting validation 

tracker = TokenTracker()

tracker.add_prompt_tokens(100)
tracker.add_completion_tokens(50)
tracker.add_retrieval_tokens(25)

tracker.summary()

# token efficiency validation 

token_efficiency(
    total_tokens=1000,
    correct_answers=5,
)

