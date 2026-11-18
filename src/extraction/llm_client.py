"""Single interface over LLM providers that return Pydantic-validated structured output.

The provider is chosen with the LLM_PROVIDER environment variable, so the rest of the
code never imports a provider SDK directly. Only Anthropic is implemented for now;
OpenAI and Ollama will be added as new classes with the same `extract` method.
"""

import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, Protocol, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output). Used only to log an estimated cost per call.
PRICES_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "claude-opus-5-5": (Decimal("4"), Decimal("20")),
    "claude-sonnet-5-5": (Decimal("2"), Decimal("10")),
    "claude-haiku-5-5": (Decimal("0.10"), Decimal("0.50")),
}


@dataclass
class LLMResult(Generic[T]):
    """Parsed output plus the usage metadata we want to track per document."""

    output: T
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float

    @property
    def cost_usd(self) -> Decimal | None:
        """Estimated cost of the call, or None if the model has no known price."""
        prices = PRICES_PER_MTOK.get(self.model)
        if prices is None:
            return None
        input_price, output_price = prices
        return (self.input_tokens * input_price + self.output_tokens * output_price) / 1_000_000


class LLMClient(Protocol):
    """What every provider must implement."""

    def extract(self, system: str, user: str, schema: type[T]) -> LLMResult[T]: ...


class AnthropicClient:
    """Claude via the official SDK, using structured outputs (`messages.parse`)."""

    def __init__(self, model: str, effort: str = "medium", max_tokens: int = 32000):
        import anthropic  # imported here so other providers don't need this package

        self._client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def extract(self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        start = time.perf_counter()
        # Streaming avoids HTTP timeouts on long outputs; we only need the final message.
        with self._client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": self.effort},
            # If a safety classifier declines, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            response = stream.get_final_message()
        seconds = time.perf_counter() - start

        if response.stop_reason != "end_turn":
            raise RuntimeError(f"LLM stopped early: stop_reason={response.stop_reason}")

        result = LLMResult(
            output=response.parsed_output,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            seconds=seconds,
        )
        logger.info(
            "%s: %d in / %d out tokens, %.1fs, ~US$ %s",
            result.model, result.input_tokens, result.output_tokens, seconds,
            f"{result.cost_usd:.4f}" if result.cost_usd is not None else "?",
        )
        return result


def get_client() -> LLMClient:
    """Build the client configured in .env (LLM_PROVIDER, LLM_MODEL)."""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-opus-5-5")
    if provider == "anthropic":
        return AnthropicClient(model=model)
    raise NotImplementedError(f"LLM provider '{provider}' is not implemented yet")
