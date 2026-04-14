import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GerritConfig:
    base_url: str
    user: str
    http_password: str

    @classmethod
    def from_env(cls) -> "GerritConfig":
        return cls(
            base_url=_require("GERRIT_URL").rstrip("/"),
            user=_require("GERRIT_USER"),
            http_password=_require("GERRIT_HTTP_PASSWORD"),
        )


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # "openai" | "anthropic"
    base_url: str | None
    api_key: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        if provider not in ("openai", "anthropic"):
            raise ValueError(f"LLM_PROVIDER must be openai|anthropic, got {provider!r}")
        return cls(
            provider=provider,
            base_url=os.environ.get("LLM_BASE_URL") or None,
            api_key=_require("LLM_API_KEY"),
            model=_require("LLM_MODEL"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        )


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"env var {name} is required")
    return v
