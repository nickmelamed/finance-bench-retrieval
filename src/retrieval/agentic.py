from __future__ import annotations

import json
from collections import Counter

from src.config.loaders import load_yaml_config
from src.llm.claude_client import ClaudeClient
from src.retrieval.base import BaseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranking import CrossEncoderReranker
from src.types.schemas import (
    AgenticConfig,
    BM25Config,
    DenseConfig,
    DocumentChunk,
    PromptTemplateConfig,
    RetrievalResult,
)
from src.utils.logging import logger


class AgenticRetriever(BaseRetriever):
    """
    Tool-using retrieval agent.

    Unlike the retrievers above, this one puts Claude in the loop:
    it decides which search tool to call, whether to expand
    neighbors or rerank, and when it has gathered enough evidence
    to stop - via an explicit `submit_evidence` tool call, bounded
    by `max_tool_calls` turns.
    """

    _reranker: CrossEncoderReranker | None = None

    def __init__(
        self,
        chunks: list[DocumentChunk],
        config: AgenticConfig,
        bm25_config: BM25Config,
        dense_config: DenseConfig | None = None,
    ):
        super().__init__(top_k=config.top_k)

        self.config = config
        self.chunks = chunks
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        self._chunk_order = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

        self.client = ClaudeClient(config.model)

        self.prompt_template = load_yaml_config(
            "prompts/agentic_search.yaml", PromptTemplateConfig
        ).template

        self.bm25 = BM25Retriever(chunks=chunks, config=bm25_config)

        self.dense = None
        self._dense_available = False

        if config.use_dense_tool and dense_config is not None:
            try:
                self.dense = DenseRetriever(config=dense_config)
                self.dense.qdrant.client.get_collections()
                self._dense_available = True
            except Exception as exc:
                logger.warning(
                    f"semantic_search tool unavailable "
                    f"(is Qdrant running?): {exc}"
                )

        self._rerank_available = config.use_rerank_tool

        self._last_usage = (0, 0)
        self._last_turn_count = 0
        self._seen: dict[str, RetrievalResult] = {}
        self._seen_frequency: Counter = Counter()

    # public

    def retrieve(self, query: str) -> list[RetrievalResult]:
        self._seen: dict[str, RetrievalResult] = {}
        self._seen_frequency: Counter = Counter()

        prompt_tokens = 0
        completion_tokens = 0

        tools = self._build_tools()

        system = [
            {
                "type": "text",
                "text": self.prompt_template.format(question=query),
                # static per question across every turn of this call
                "cache_control": {"type": "ephemeral"},
            }
        ]

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Begin gathering evidence."}],
            }
        ]

        tool_result_positions: list[int] = []

        final_chunk_ids: list[str] | None = None

        for turn in range(self.config.max_tool_calls):
            if turn == self.config.max_tool_calls - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This is your final turn. You must call "
                                    "submit_evidence now with the best "
                                    "chunk_ids you have found so far."
                                ),
                            }
                        ],
                    }
                )

            self._mark_cache_breakpoint(messages)

            response = self.client.generate_with_tools(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=self.config.max_tokens_per_turn,
            )

            usage = response.usage

            prompt_tokens += (
                usage.input_tokens
                + getattr(usage, "cache_creation_input_tokens", 0)
                + getattr(usage, "cache_read_input_tokens", 0)
            )
            completion_tokens += usage.output_tokens

            messages.append(
                {
                    "role": "assistant",
                    # normalize to plain dicts so cache_control/trimming
                    # can mutate blocks uniformly regardless of source
                    "content": [block.model_dump() for block in response.content],
                }
            )

            if response.stop_reason != "tool_use":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Please call a tool, or call "
                                    "submit_evidence with your final "
                                    "chunk_ids."
                                ),
                            }
                        ],
                    }
                )
                continue

            submit_block = None
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name == "submit_evidence":
                    submit_block = block
                    continue

                output = self._execute_tool(block.name, block.input)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output),
                    }
                )

            if submit_block is not None:
                final_chunk_ids = self._validate_chunk_ids(
                    submit_block.input.get("chunk_ids", [])
                )
                break

            messages.append({"role": "user", "content": tool_results})
            tool_result_positions.append(len(messages) - 1)
            self._collapse_stale_tool_results(messages, tool_result_positions)

        if final_chunk_ids is None:
            logger.warning(
                "Agentic retriever hit max_tool_calls without "
                "submit_evidence; falling back to best-seen chunks."
            )
            final_chunk_ids = self._fallback_chunk_ids()

        self._last_usage = (prompt_tokens, completion_tokens)
        self._last_turn_count = turn + 1

        return self._build_results(final_chunk_ids)

    def last_retrieval_usage(self) -> tuple[int, int]:
        return self._last_usage

    def last_retrieval_turn_count(self) -> int:
        return self._last_turn_count

    # context management (prompt caching + trimming)

    def _mark_cache_breakpoint(self, messages: list[dict]) -> None:
        """
        Move the ephemeral cache breakpoint to the last content block of
        the last message. Anthropic caches everything up to and including
        a marked block; since `messages` only ever grows by appending,
        each turn's prefix matches the previous turn's cached prefix, so
        only the newly-appended tail is charged as fresh input.
        """
        for msg in messages:
            for block in msg["content"]:
                block.pop("cache_control", None)

        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    def _collapse_stale_tool_results(
        self, messages: list[dict], tool_result_positions: list[int]
    ) -> None:
        """
        Shrink tool_result payloads once they age past the most recent
        `keep_recent_tool_turns` - the agent already has their chunk_ids
        in `self._seen`, so re-sending full chunk text on every
        subsequent turn just inflates cost without adding information.
        """
        keep = self.config.keep_recent_tool_turns

        if len(tool_result_positions) <= keep:
            return

        stale_idx = tool_result_positions[-(keep + 1)]
        message = messages[stale_idx]

        for block in message["content"]:
            if block.get("type") != "tool_result":
                continue

            try:
                parsed = json.loads(block["content"])
                chunk_ids = [
                    item["chunk_id"]
                    for item in parsed
                    if isinstance(item, dict) and "chunk_id" in item
                ]
            except Exception:
                chunk_ids = []

            block["content"] = json.dumps(
                {
                    "note": "earlier result, full text omitted to save context",
                    "chunk_ids": chunk_ids,
                }
            )

    # tool schema

    def _build_tools(self) -> list[dict]:
        tools = [
            {
                "name": "lexical_search",
                "description": (
                    "Keyword/exact-match search over the corpus. Best "
                    "for exact figures, line-item names, dates."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            }
        ]

        if self._dense_available:
            tools.append(
                {
                    "name": "semantic_search",
                    "description": (
                        "Meaning-based search over the corpus. Best for "
                        "paraphrased or conceptual questions."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                }
            )

        tools.append(
            {
                "name": "get_neighbors",
                "description": (
                    "Fetch chunks adjacent to a chunk you've already "
                    "found, in case a table or statement continues "
                    "past its edges."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "window": {"type": "integer"},
                    },
                    "required": ["chunk_id"],
                },
            }
        )

        if self._rerank_available:
            tools.append(
                {
                    "name": "rerank",
                    "description": (
                        "Re-score a broader candidate set of chunk_ids "
                        "against the question using a cross-encoder, "
                        "for a precision pass over noisy search results."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "chunk_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query", "chunk_ids"],
                    },
                }
            )

        tools.append(
            {
                "name": "submit_evidence",
                "description": (
                    "Call this when you have gathered enough evidence "
                    "chunks to answer the question. Ends the search."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "chunk_ids you have actually seen "
                                "returned by a tool in this conversation"
                            ),
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": ["chunk_ids", "reasoning"],
                },
            }
        )

        # tools are identical across every turn and every question
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

        return tools

    # tool execution

    def _execute_tool(self, name: str, tool_input: dict) -> list[dict] | dict:
        if name == "lexical_search":
            top_k = tool_input.get("top_k") or self.config.search_top_k
            results = self.bm25.retrieve(tool_input["query"])[:top_k]

        elif name == "semantic_search" and self._dense_available:
            top_k = tool_input.get("top_k") or self.config.search_top_k
            results = self.dense.retrieve(tool_input["query"])[:top_k]

        elif name == "get_neighbors":
            results = self._get_neighbors(
                tool_input["chunk_id"], tool_input.get("window")
            )

        elif name == "rerank" and self._rerank_available:
            results = self._rerank(
                tool_input["query"], tool_input.get("chunk_ids", [])
            )

        else:
            return {"error": f"unknown or unavailable tool: {name}"}

        self._record_seen(results)

        preview_chars = self.config.tool_result_preview_chars

        return [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.metadata.get("document_id", ""),
                "score": round(r.score, 4),
                "text": r.text[:preview_chars],
            }
            for r in results
        ]

    def _get_neighbors(
        self, chunk_id: str, window: int | None
    ) -> list[RetrievalResult]:
        window = window if window is not None else self.config.neighbor_window

        idx = self._chunk_order.get(chunk_id)

        if idx is None:
            return []

        # inherit a fraction of the anchor's score rather than 0.0, so
        # neighbors the agent deliberately sought out aren't unfairly
        # discounted against generic search hits in the fallback ranking
        anchor = self._seen.get(chunk_id)
        neighbor_score = anchor.score * 0.75 if anchor else 0.0

        neighbors = []

        for offset in range(-window, window + 1):
            if offset == 0:
                continue

            n_idx = idx + offset

            if n_idx < 0 or n_idx >= len(self.chunks):
                continue

            chunk = self.chunks[n_idx]

            neighbors.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=neighbor_score,
                    retrieval_method="agentic",
                    metadata={"document_id": chunk.document_id, **chunk.metadata},
                )
            )

        return neighbors

    def _rerank(self, query: str, chunk_ids: list) -> list[RetrievalResult]:
        candidates = []

        for cid in chunk_ids:
            chunk = self.chunk_lookup.get(cid)

            if chunk is None:
                continue

            candidates.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=0.0,
                    retrieval_method="agentic",
                    metadata={"document_id": chunk.document_id, **chunk.metadata},
                )
            )

        if not candidates:
            return []

        # cap output like the search tools; reranking a big candidate
        # set shouldn't mean paying to resend every candidate's full text
        top_k = min(len(candidates), self.config.search_top_k)

        return self._get_reranker().rerank(query, candidates, top_k=top_k)

    def _get_reranker(self) -> CrossEncoderReranker:
        if AgenticRetriever._reranker is None:
            AgenticRetriever._reranker = CrossEncoderReranker()

        return AgenticRetriever._reranker

    # bookkeeping

    def _record_seen(self, results: list[RetrievalResult]) -> None:
        for r in results:
            self._seen_frequency[r.chunk_id] += 1

            existing = self._seen.get(r.chunk_id)

            if existing is None or r.score > existing.score:
                self._seen[r.chunk_id] = r

    def _validate_chunk_ids(self, chunk_ids: list) -> list[str]:
        valid = [
            cid
            for cid in chunk_ids
            if isinstance(cid, str) and cid in self._seen
        ]

        dropped = [cid for cid in chunk_ids if cid not in valid]

        if dropped:
            logger.warning(
                f"Agentic retriever dropped hallucinated chunk_ids "
                f"not seen via tools: {dropped}"
            )

        if not valid:
            return self._fallback_chunk_ids()

        return valid[: self.top_k]

    def _fallback_chunk_ids(self) -> list[str]:
        ranked = sorted(
            self._seen.keys(),
            key=lambda cid: (self._seen_frequency[cid], self._seen[cid].score),
            reverse=True,
        )

        return ranked[: self.top_k]

    def _build_results(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        results = []

        for cid in chunk_ids[: self.top_k]:
            chunk = self.chunk_lookup.get(cid)

            if chunk is None:
                continue

            seen_result = self._seen.get(cid)

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=seen_result.score if seen_result else 0.0,
                    retrieval_method="agentic",
                    metadata={"document_id": chunk.document_id, **chunk.metadata},
                )
            )

        return results
