"""The optional FastAPI backend: importable, and every route/schema builds into OpenAPI."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")


def test_api_app_builds_openapi_schema():
    from semg_pp_ml.api.main import app
    schema = app.openapi()
    assert schema["paths"], "no routes registered"
    assert any("/artifacts" in p for p in schema["paths"])


def test_register_login_and_protected_route():
    from fastapi.testclient import TestClient
    from semg_pp_ml.api.main import app
    email = "tester@example.com"
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/v1/models/").status_code == 401          # no token
        r = client.post("/api/v1/auth/register",
                        json={"email": email, "password": "correct-horse-1", "full_name": "Tester"})
        assert r.status_code in (201, 400)                                # 400 if already registered
        r = client.post("/api/v1/auth/login", json={"email": email, "password": "correct-horse-1"})
        assert r.status_code == 200
        token = r.json()["access_token"]
        assert client.get("/api/v1/models/", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        assert client.post("/api/v1/auth/login",
                           json={"email": email, "password": "wrong-password"}).status_code == 401
