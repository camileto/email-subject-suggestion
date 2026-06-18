from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_PAYLOAD = {
    "customer": {"user_id": "123", "name": "Maria"},
    "products": [{"name": "Tênis Runner", "price_full": 300, "price_promo": 240}],
}


def test_unconfigured_provider_returns_400_not_500(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.clients.anthropic_client import get_client as anthropic_get_client

    anthropic_get_client.cache_clear()
    with patch("app.main.get_client"):
        response = client.post(
            "/api/v1/subjects", json={**_PAYLOAD, "provider": "anthropic"}
        )
    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_default_provider_is_openai_and_missing_key_is_400_not_500(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app.clients.openai_client import get_client as openai_get_client

    openai_get_client.cache_clear()
    response = client.post("/api/v1/subjects", json=_PAYLOAD)
    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.json()["detail"]
