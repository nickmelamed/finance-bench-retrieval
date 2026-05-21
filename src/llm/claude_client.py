import os
from anthropic import Anthropic
from dotenv import load_dotenv

from src.llm.caching import cache
from src.utils.logging import logger


load_dotenv()


class ClaudeClient:
    def __init__(self, model: str):
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512):
        cache_key = f"{self.model}:{hash(prompt)}"

        if cache_key in cache:
            return cache[cache_key]

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

        cache[cache_key] = response

        logger.info("Claude response generated")

        return response