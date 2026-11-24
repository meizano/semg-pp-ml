"""The optional FastAPI backend: importable, and every route/schema builds into OpenAPI."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")


def test_api_app_builds_openapi_schema():
    from semg_pp_ml.api.main import app
    schema = app.openapi()
    assert schema["paths"], "no routes registered"
    assert any("/artifacts" in p for p in schema["paths"])
