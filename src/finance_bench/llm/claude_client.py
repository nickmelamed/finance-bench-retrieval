import hashlib
import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

from finance_bench.llm.caching import cache
from finance_bench.utils.logging import logger


load_dotenv()

ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0}


class ClaudeClient:
    def __init__(self, model: str):
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model

    def _cache_key(self, prompt: str, max_tokens: int) -> str:
        prompt_hash = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()

        return f"{self.model}:{max_tokens}:{prompt_hash}"

    def generate(self, prompt: str, max_tokens: int = 512) -> dict:
        """
        Single-shot cached call. Returns {"text": str, "usage": dict}.
        A cache hit costs nothing, so its usage is reported as zero -
        it did no fresh work and shouldn't be double-counted against
        the token-efficiency metric.
        """
        cache_key = self._cache_key(prompt, max_tokens)

        if cache_key in cache:
            return {"text": cache[cache_key]["text"], "usage": dict(ZERO_USAGE)}

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text = response.content[0].text

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        cache[cache_key] = {"text": text}

        logger.info("Claude response generated")

        return {"text": text, "usage": usage}

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        poll_interval: float = 10.0,
        max_wait_seconds: float = 3600.0,
    ) -> list[dict]:
        """
        Submit every not-already-cached prompt as ONE Anthropic
        Message Batch (50% cheaper than the same calls made
        synchronously), poll until it finishes, and return results
        in the same order as `prompts`. Each item is
        {"text": str, "usage": dict}; cache hits report zero usage,
        same convention as `generate`.
        """
        cache_keys = [self._cache_key(p, max_tokens) for p in prompts]

        results: list[dict | None] = [None] * len(prompts)
        to_submit: list[tuple[int, str]] = []

        for i, (prompt, key) in enumerate(zip(prompts, cache_keys)):
            if key in cache:
                results[i] = {"text": cache[key]["text"], "usage": dict(ZERO_USAGE)}
            else:
                to_submit.append((i, prompt))

        if not to_submit:
            return results

        requests = [
            {
                "custom_id": str(i),
                "params": {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
            }
            for i, prompt in to_submit
        ]

        batch = self.client.messages.batches.create(requests=requests)

        logger.info(
            f"Submitted batch {batch.id} with {len(requests)} requests "
            f"({len(prompts) - len(to_submit)} served from cache)"
        )

        waited = 0.0

        while batch.processing_status != "ended":
            if waited >= max_wait_seconds:
                raise TimeoutError(
                    f"Batch {batch.id} did not finish within "
                    f"{max_wait_seconds}s (status={batch.processing_status})"
                )

            time.sleep(poll_interval)
            waited += poll_interval

            batch = self.client.messages.batches.retrieve(batch.id)

        logger.info(f"Batch {batch.id} complete: {batch.request_counts}")

        for entry in self.client.messages.batches.results(batch.id):
            idx = int(entry.custom_id)
            key = cache_keys[idx]

            if entry.result.type == "succeeded":
                message = entry.result.message

                text = message.content[0].text if message.content else ""

                usage = {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }

                cache[key] = {"text": text}

                results[idx] = {"text": text, "usage": usage}

            else:
                logger.warning(
                    f"Batch item {idx} did not succeed: "
                    f"{entry.result.type}"
                )

                results[idx] = {"text": "", "usage": dict(ZERO_USAGE)}

        return results

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
        max_tokens: int = 1024,
    ):
        """
        Uncached multi-turn tool-use call. Multi-turn tool
        conversations aren't a good fit for the single-prompt
        cache in `generate` (each call depends on the full
        conversation state, not just a static prompt string).
        """

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=system,
            tools=tools,
            messages=messages,
        )

        logger.info(
            f"Claude tool-use turn generated "
            f"(stop_reason={response.stop_reason})"
        )

        return response