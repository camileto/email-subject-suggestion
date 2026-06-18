from unittest.mock import MagicMock, patch

import pytest

from app.clients.llm_client import RawVariant, RawVariantList, generate_variants

_MESSAGES = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "user prompt"},
]

_VARIANTS = [RawVariant(subject="s", preheader="p", trigger="curiosity", rationale="r")]


def test_generate_variants_openai_uses_parse_and_returns_variants():
    fake_client = MagicMock()
    fake_client.beta.chat.completions.parse.return_value.choices = [
        MagicMock(message=MagicMock(parsed=RawVariantList(variants=_VARIANTS)))
    ]
    with patch("app.clients.openai_client.get_client", return_value=fake_client):
        result = generate_variants("openai", _MESSAGES, model="gpt-4o-mini")
    assert result == _VARIANTS


def test_generate_variants_anthropic_splits_system_and_user():
    fake_client = MagicMock()
    fake_client.messages.parse.return_value.parsed_output = RawVariantList(variants=_VARIANTS)
    with patch("app.clients.anthropic_client.get_client", return_value=fake_client):
        result = generate_variants("anthropic", _MESSAGES, model="claude-opus-4-8")
    assert result == _VARIANTS
    _, kwargs = fake_client.messages.parse.call_args
    assert kwargs["system"] == "system prompt"
    assert kwargs["messages"] == [{"role": "user", "content": "user prompt"}]


def test_generate_variants_gemini_uses_response_schema():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value.parsed = RawVariantList(variants=_VARIANTS)
    with patch("app.clients.gemini_client.get_client", return_value=fake_client):
        result = generate_variants("gemini", _MESSAGES, model="gemini-2.5-flash")
    assert result == _VARIANTS


def test_anthropic_client_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.clients import anthropic_client

    anthropic_client.get_client.cache_clear()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_client.get_client()


def test_gemini_client_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from app.clients import gemini_client

    gemini_client.get_client.cache_clear()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_client.get_client()
