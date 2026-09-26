from fastapi.testclient import TestClient

from layajev.main import app


class FakeRouter:
    def predict(self, state, questions, **options):
        return {"answers": {"billing": {"noul": 0.9}}, "state": state, "options": options}


def test_system_one_returns_router_prediction(monkeypatch):
    monkeypatch.setattr("layajev.main.get_router", lambda: FakeRouter())
    client = TestClient(app)

    response = client.post(
        "/v1/systemone",
        json={
            "state": {"document": "Please refund the duplicate charge."},
            "questions": {"billing": {"type": "noul", "instructions": "Is this billing?"}},
            "model": "multilingual",
        },
    )

    assert response.status_code == 200
    assert response.json()["answers"]["billing"]["noul"] == 0.9
    assert response.json()["options"] == {"model": "multilingual"}
    assert float(response.headers["X-Process-Time-MS"]) >= 0


def test_system_one_requires_configured_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr("layajev.main.get_router", lambda: FakeRouter())
    client = TestClient(app)
    payload = {"state": "hello", "questions": {"q": {"type": "noul", "instructions": "yes?"}}}

    assert client.post("/v1/systemone", json=payload).status_code == 401
    assert client.post(
        "/v1/systemone", json=payload, headers={"Authorization": "Bearer test-key"}
    ).status_code == 200


def test_system_one_accepts_legacy_laya_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("LAYA_API_KEY", "legacy-key")
    monkeypatch.setattr("layajev.main.get_router", lambda: FakeRouter())
    client = TestClient(app)
    payload = {"state": "hello", "questions": {"q": {"type": "noul", "instructions": "yes?"}}}

    assert client.post(
        "/v1/systemone", json=payload, headers={"Authorization": "Bearer legacy-key"}
    ).status_code == 200


def test_health_and_console_do_not_load_model(monkeypatch):
    monkeypatch.setattr("layajev.main.get_router", lambda: (_ for _ in ()).throw(AssertionError()))
    client = TestClient(app)

    assert client.get("/healthz").json() == {"status": "ok"}
    assert "LayaJev" in client.get("/").text