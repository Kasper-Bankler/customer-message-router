"""LLMClient interface plus an Ollama implementation; every LLM call in the system goes through here.

The interface exists before there are two implementations, which the complexity
budget otherwise forbids. The stated reason is concentration risk: a bank cannot
have provider SDK calls scattered through its agents, because swapping provider,
adding a cost cap or routing EU traffic then means editing every agent. One
chokepoint makes those one-file changes.
"""

import json
from pathlib import Path
from typing import TypeVar

import ollama
import yaml
from pydantic import BaseModel, ValidationError

from src.schemas import TokenCost

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_YAML = REPO_ROOT / "config" / "settings.yaml"

_settings = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8"))["llm"]
MODEL_NAME: str = _settings["model"]
TEMPERATURE: float = _settings["temperature"]
TIMEOUT_SECONDS: int = _settings["timeout_seconds"]
MAX_VALIDATION_RETRIES: int = _settings["max_validation_retries"]

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or could not produce output matching the schema.

    Callers must treat this as a routing failure and fail closed to HUMAN. It is
    never a reason to guess at what the customer wanted.
    """


class LLMClient:
    """One method: prompt in, validated Pydantic object out.

    Deliberately not an ABC. A plain base class raising NotImplementedError is the
    same contract with none of the metaclass machinery to explain.
    """

    def complete_structured(self, prompt: str, schema: type[ModelT]) -> tuple[ModelT, TokenCost]:
        """Return an instance of `schema`, plus what the call cost."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Local Ollama. No customer data leaves the machine, which is the point."""

    def __init__(self, model: str = MODEL_NAME, temperature: float = TEMPERATURE) -> None:
        self.model = model
        self.temperature = temperature
        self.client = ollama.Client(timeout=TIMEOUT_SECONDS)

    def complete_structured(self, prompt: str, schema: type[ModelT]) -> tuple[ModelT, TokenCost]:
        """Ask for JSON matching `schema`, validate it, retry once on failure.

        Ollama is given the schema itself as its `format`, so the model is
        constrained during decoding rather than merely asked nicely. Validation
        still runs, because a constrained decoder can still emit a wrong-typed or
        semantically empty object.
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        cost = TokenCost()
        last_error: Exception | None = None

        for _ in range(MAX_VALIDATION_RETRIES + 1):
            content, call_cost = self._chat(messages, schema)
            cost = _add_cost(cost, call_cost)
            try:
                return schema.model_validate_json(content), cost
            except ValidationError as error:
                last_error = error
                # Show the model its own output and the specific complaint. A bare
                # "try again" reliably produces the same invalid output again.
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That failed validation: {error}. Return only JSON matching the schema."},
                ]

        raise LLMUnavailable(f"{self.model} did not return valid {schema.__name__}") from last_error

    def _chat(self, messages: list[dict[str, str]], schema: type[ModelT]) -> tuple[str, TokenCost]:
        """One round trip. Wraps transport failures in LLMUnavailable so callers fail closed."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=schema.model_json_schema(),
                options={"temperature": self.temperature},
            )
        except (ollama.ResponseError, ConnectionError, TimeoutError) as error:
            raise LLMUnavailable(f"could not reach Ollama model {self.model}: {error}") from error

        return response.message.content or "", TokenCost(
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
            # Local inference has no per-token price. Kept at zero rather than
            # invented, so any non-zero cost in a trace means a hosted call.
            usd=0.0,
        )


def _add_cost(left: TokenCost, right: TokenCost) -> TokenCost:
    """Accumulate spend across retries, so a retried call reports what it really cost."""
    return TokenCost(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        usd=left.usd + right.usd,
    )
