from google.genai import types
from pydantic import BaseModel

from . import anthropic_client, gemini_client, openai_client


class RawVariant(BaseModel):
    subject: str
    preheader: str
    trigger: str
    rationale: str


class RawVariantList(BaseModel):
    variants: list[RawVariant]


def _split_system_and_user(messages: list[dict]) -> tuple[str, str]:
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")
    return system, user


def _generate_openai(messages: list[dict], model: str) -> list[RawVariant]:
    """Uses OpenAI structured outputs (response_format=pydantic model) so the
    response is guaranteed to match the schema — no manual/defensive JSON
    parsing needed on the way out."""
    client = openai_client.get_client()
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=RawVariantList,
        temperature=0.9,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("The model did not return structured output")
    return parsed.variants


def _generate_anthropic(messages: list[dict], model: str) -> list[RawVariant]:
    client = anthropic_client.get_client()
    system, user = _split_system_and_user(messages)
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=RawVariantList,
    )
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("The model did not return structured output")
    return parsed.variants


def _generate_gemini(messages: list[dict], model: str) -> list[RawVariant]:
    client = gemini_client.get_client()
    system, user = _split_system_and_user(messages)
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=RawVariantList,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        raise RuntimeError("The model did not return structured output")
    return parsed.variants


_GENERATORS = {
    "openai": _generate_openai,
    "anthropic": _generate_anthropic,
    "gemini": _generate_gemini,
}


def generate_variants(provider: str, messages: list[dict], model: str) -> list[RawVariant]:
    return _GENERATORS[provider](messages, model)
