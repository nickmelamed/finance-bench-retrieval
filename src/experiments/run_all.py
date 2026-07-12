from __future__ import annotations

import json

from dotenv import load_dotenv

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.agentic import AgenticRetriever

from src.evaluation.qa_metrics import (
    accuracy,
    token_efficiency,
)

from src.evaluation.retrieval_metrics import (
    recall_at_k,
    hit_rate,
    mean_reciprocal_rank,
)

from src.evaluation.bootstrap import (
    BootstrapCI,
)

from src.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)

from src.evaluation.experiment_tracker import (
    ExperimentTracker,
)

from src.visualizations.plots import (
    PlotGenerator,
)

from src.config.loaders import load_yaml_config

from src.types.schemas import (
    ExperimentConfig,
    BM25Config,
    DenseConfig,
    HybridConfig,
    AgenticConfig,
    DocumentChunk,
    PromptTemplateConfig
)


load_dotenv()


# configs

config = load_yaml_config(
    "experiment.yaml",
    ExperimentConfig,
)

bm25_config = load_yaml_config(
    "retrieval/bm25.yaml",
    BM25Config,
)

dense_config = load_yaml_config(
    "retrieval/dense.yaml",
    DenseConfig,
)

hybrid_config = load_yaml_config(
    "retrieval/hybrid.yaml",
    HybridConfig,
)

agentic_config = load_yaml_config(
    "retrieval/agentic.yaml",
    AgenticConfig,
)

bootstrap_config = config.bootstrap

judge_prompt = load_yaml_config(
    "prompts/correctness.yaml",
    PromptTemplateConfig
).template

qa_prompt = load_yaml_config(
    "prompts/qa.yaml",
    PromptTemplateConfig
).template


MODEL = config.llm.model


# data loading

def load_questions() -> list[dict]:

    with open(
        "data/processed/financebench_examples.json",
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_chunks() -> list[DocumentChunk]:

    with open(
        "data/processed/chunks.json",
        "r",
        encoding="utf-8",
    ) as f:

        raw_chunks = json.load(f)

    return [
        DocumentChunk.model_validate(chunk)
        for chunk in raw_chunks
    ]


# evaluation

def evaluate_retrieval(
    name: str,
    retriever,
    questions: list[dict],
):

    recalls = []

    hit_rates = []

    mrrs = []

    total_questions = len(questions)

    print(
        f"\nEvaluating retrieval "
        f"{name} on "
        f"{total_questions} questions..."
    )

    for idx, item in enumerate(questions):

        question = item["question"]

        gold_chunk_ids = item.get(
            "gold_chunk_ids",
            [],
        )

        if not gold_chunk_ids:
            continue 

        retrieved = retriever.retrieve(
            question
        )

        retrieved_ids = [
            r.chunk_id
            for r in retrieved
        ]

        recalls.append(
            recall_at_k(
                retrieved_ids,
                gold_chunk_ids,
            )
        )

        hit_rates.append(
            hit_rate(
                retrieved_ids,
                gold_chunk_ids,
            )
        )

        mrrs.append(
            mean_reciprocal_rank(
                retrieved_ids,
                gold_chunk_ids,
            )
        )

        if (idx + 1) % 10 == 0:

            print(
                f"[{name}] "
                f"{idx + 1}/"
                f"{total_questions}"
            )

    return {
        "method": name,
        "recall_at_k": (
            sum(recalls)
            / len(recalls)
            if recalls else 0.0
        ),
        "hit_rate": (
            sum(hit_rates)
            / len(hit_rates)
            if hit_rates else 0.0 
        ),
        "mrr": (
            sum(mrrs)
            / len(mrrs)

            if mrrs else 0.0 
        ),
    }

def evaluate_qa(
    name: str,
    retriever,
    pipeline: EvaluationPipeline,
    questions: list[dict],
):

    total_questions = len(questions)

    print(
        f"\nEvaluating QA "
        f"{name} on "
        f"{total_questions} questions..."
    )

    # phase 1: retrieval - sequential and unavoidable, since a
    # multi-turn tool-use conversation (agentic) can't be batched:
    # each turn depends on the previous turn's result.

    retrieved_per_question = []

    retrieval_usage_per_question = []

    for idx, item in enumerate(questions):

        retrieved = retriever.retrieve(
            item["question"]
        )

        retrieval_prompt_tokens, retrieval_completion_tokens = (
            retriever.last_retrieval_usage()
        )

        pipeline.token_tracker.add_prompt_tokens(
            retrieval_prompt_tokens
        )

        pipeline.token_tracker.add_completion_tokens(
            retrieval_completion_tokens
        )

        retrieved_per_question.append(retrieved)

        retrieval_usage_per_question.append(
            (
                retrieval_prompt_tokens,
                retrieval_completion_tokens,
                retriever.last_retrieval_turn_count(),
            )
        )

        if (idx + 1) % 10 == 0:

            print(
                f"[{name}] retrieval "
                f"{idx + 1}/{total_questions}"
            )

    # phase 2: batch QA generation - one Anthropic Message Batch for
    # all {total_questions} prompts instead of {total_questions}
    # synchronous calls (50% cheaper; no per-turn dependency here,
    # every question's answer is independent of every other's).

    print(
        f"\n[{name}] submitting QA-generation batch "
        f"({total_questions} questions)..."
    )

    contexts = [
        pipeline.build_context(retrieved)
        for retrieved in retrieved_per_question
    ]

    questions_and_contexts = [
        (item["question"], context)
        for item, context in zip(questions, contexts)
    ]

    generated_answers = pipeline.answer_batch(
        questions_and_contexts
    )

    print(f"[{name}] QA-generation batch complete.")

    # phase 3: batch grading - deterministic matches never touch the
    # API; only ambiguous answers are batched to the LLM judge.

    print(f"[{name}] grading...")

    grading_items = [
        {
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "generated_answer": answer,
        }
        for item, answer in zip(questions, generated_answers)
    ]

    gradings_and_usage = pipeline.grader.grade_batch(
        grading_items
    )

    correctness = []

    per_question = []

    for item, generated_answer, retrieved, retrieval_usage, (
        grading,
        judge_usage,
    ) in zip(
        questions,
        generated_answers,
        retrieved_per_question,
        retrieval_usage_per_question,
        gradings_and_usage,
    ):

        pipeline.token_tracker.add_prompt_tokens(
            judge_usage["input_tokens"]
        )

        pipeline.token_tracker.add_completion_tokens(
            judge_usage["output_tokens"]
        )

        is_correct = int(grading["correct"])

        correctness.append(is_correct)

        if not is_correct:

            pipeline.failure_analyzer.classify(
                question=item["question"],
                generated_answer=generated_answer,
                gold_answer=item["gold_answer"],
                retrieved_chunks=retrieved,
            )

        retrieval_prompt_tokens, retrieval_completion_tokens, retrieval_turns = (
            retrieval_usage
        )

        per_question.append(
            {
                "question_id": item.get("question_id"),
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "generated_answer": generated_answer,
                "correct": bool(is_correct),
                "retrieval_prompt_tokens": retrieval_prompt_tokens,
                "retrieval_completion_tokens": retrieval_completion_tokens,
                "retrieval_turns": retrieval_turns,
                "retrieved_chunk_ids": [
                    r.chunk_id for r in retrieved
                ],
            }
        )

    running_acc = (
        sum(correctness) / len(correctness)
        if correctness else 0.0
    )

    print(f"[{name}] Accuracy: {running_acc:.4f}")

    tokens = (
        pipeline
        .token_tracker
        .total_tokens
    )

    correct_answers = sum(correctness)

    return {
        "method": name,
        "accuracy": (
            accuracy(correctness)
        ),
        "total_tokens": tokens,
        "token_efficiency": token_efficiency(
        total_tokens=tokens,
        correct_answers=correct_answers,
        ), 
        "correctness": correctness,
        "failures": (
            pipeline.failure_analyzer.summary()
        ),
        "per_question": per_question,
    }


# main 

def main():

    print(
        "\n=== FinanceBench Evaluation ===\n"
    )

    questions = load_questions()

    chunks = load_chunks()

    print(
        f"Loaded {len(questions)} questions"
    )

    print(
        f"Loaded {len(chunks)} chunks"
    )

    tracker = ExperimentTracker()

    retrievers = {

        "bm25": BM25Retriever(
            chunks=chunks,
            config=bm25_config,
        ),

        "dense": DenseRetriever(
            config=dense_config,
        ),

        "hybrid": HybridRetriever(
            chunks=chunks,
            hybrid_config=hybrid_config,
            bm25_config=bm25_config,
            dense_config=dense_config,
        ),

        "agentic": AgenticRetriever(
            chunks=chunks,
            config=agentic_config,
            bm25_config=bm25_config,
            dense_config=dense_config,
        ),
    }

    results = []

    bootstrap = BootstrapCI(bootstrap_config)

    for name, retriever in retrievers.items():

        pipeline = EvaluationPipeline(
            MODEL,
            judge_prompt,
            qa_prompt,
        )

        retrieval_result = (
            evaluate_retrieval(
                name=name,
                retriever=retriever,
                questions=questions,
            )
        )

        qa_result = evaluate_qa(
            name=name,
            retriever=retriever,
            pipeline=pipeline,
            questions=questions,
        )

        ci = bootstrap.compute(
            qa_result["correctness"]
        )

        combined = {
            **retrieval_result,
            **qa_result,
            "bootstrap_ci": ci
        }

        results.append(combined)

        # save after every method so a later method crashing  
        # shouldn't lose already-computed results for the ones that finished
        tracker.save_results(
            "evaluation_results.json",
            results,
        )

    # visualizations 

    plotter = PlotGenerator()

    methods = [
        r["method"]
        for r in results
    ]

    accuracies = [
        r["accuracy"]
        for r in results
    ]

    lower = [
        r["bootstrap_ci"]["lower"]
        for r in results
    ]

    upper = [
        r["bootstrap_ci"]["upper"]
        for r in results
    ]

    efficiencies = [
        r["token_efficiency"]
        for r in results
    ]

    plotter.accuracy_plot(
        methods,
        accuracies,
        lower,
        upper,
    )

    plotter.pareto_plot(
        methods,
        accuracies,
        efficiencies,
    )

    failure_results = {
    result["method"]: result["failures"]
    for result in results
    }

    plotter.failure_plot(
        failure_results
    )

    print("\n=== FINAL RESULTS ===\n")

    for result in results:

        print(
            f"{result['method']:10s} "
            f"| Accuracy: "
            f"{result['accuracy']:.4f} "
            f"| Tokens: "
            f"{result['total_tokens']}"
        )

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()