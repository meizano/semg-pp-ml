# REST API

The API is an optional FastAPI backend. It exposes preprocessing, training, grid search and artifact management over HTTP, with user accounts and JWT login. Use it when another program or a remote client needs to drive the platform. For normal work, the [CLI](cli.md) and the [web app](web-app.md) are simpler.

> **Status.** The API imports, its OpenAPI schema builds and it starts. Its job flows and authentication have not been tested end to end in this repository yet. Treat it as experimental. See the security notes below before exposing it to others.

## Install and start

```bash
pip install -e ".[api,dl]"
uvicorn semg_pp_ml.api.main:app --reload --port 8000
```

- Interactive documentation: `http://localhost:8000/docs` (Swagger UI).
- Health check: `GET /health`.
- Database: SQLite file `semg_api.db` in the folder you start from. Set `DATABASE_URL` for PostgreSQL, for example `postgresql+psycopg://user:password@localhost:5432/semg_ml` (install `psycopg` too).

Background jobs run inside the API process (FastAPI background tasks). No Redis or Celery worker is needed for the default setup.

## Authentication

1. Register: `POST /api/v1/auth/register` with `{"email": "...", "password": "at least 8 characters", "full_name": "optional"}`.
2. Log in: `POST /api/v1/auth/login` with `{"email": "...", "password": "..."}`. The response contains an access token and a refresh token.
3. Send the token on every other request: `Authorization: Bearer <access_token>`.
4. Refresh: `POST /api/v1/auth/refresh` with the refresh token.
5. Your profile: `GET` and `PUT /api/v1/auth/me`.

Example with curl:

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "your-password"}'
```

## Endpoints

| Area | Method and path | Purpose |
|---|---|---|
| System | `GET /health`, `GET /api/v1/system` | status and system information |
| Config | `GET`, `POST /api/v1/config` | read or set the platform configuration |
| Preprocessing | `GET /api/v1/preprocessing/methods` | list methods |
| | `POST /api/v1/preprocessing/run` | start a preprocessing job |
| | `GET /api/v1/preprocessing/jobs`, `GET`/`DELETE /api/v1/preprocessing/jobs/{job_id}` | list, check or delete jobs |
| | `GET /api/v1/preprocessing/jobs/{job_id}/result` | fetch the result |
| Models | `GET /api/v1/models/`, `GET /api/v1/models/categories` | list models |
| | `GET /api/v1/models/{model_type}/hyperparameters` | hyperparameters of one model |
| | `POST /api/v1/models/train` | start a training job |
| | `GET /api/v1/models/jobs`, `GET /api/v1/models/jobs/{job_id}`, `.../result` | follow training jobs |
| Experiments | `POST /api/v1/experiments/grid-search` | start a grid search |
| | `GET /api/v1/experiments/`, `GET`/`DELETE /api/v1/experiments/{experiment_id}` | list and manage experiments |
| | `GET /api/v1/experiments/{experiment_id}/status` | progress |
| Artifacts | `GET /api/v1/artifacts/`, `GET`/`DELETE /api/v1/artifacts/{artifact_id}` | list and manage saved files |
| | `GET /api/v1/artifacts/{artifact_id}/download` | download a file |
| | `POST /api/v1/artifacts/{artifact_id}/cleanup` | remove the stored file |
| WebSocket | `/api/v1/ws` | live job updates |

The request and response fields are listed in the Swagger page at `/docs`. The main request bodies are:

- `PreprocessRequest`: `config` (preprocessing settings), `force_recompute`, `save_visualizations`.
- `TrainRequest`: `model_type`, `hyperparameters`, optional `preprocessing_job_id`, `save_model`, `skip_visualization`.
- `GridSearchRequest`: `preps`, `models`, `exclude_pairs`, `skip_visualization`, `max_parallel`.

## Security notes (read before deploying)

- **Set a secret key.** Tokens are signed with `SEMG_SECRET_KEY`. If it is not set, a development-only default is used, and anyone who knows that default can forge login tokens. Set it to a long random value before the API is reachable by anyone else, for example `export SEMG_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")`.
- **Use HTTPS** and a real database in production. SQLite is for local use.
- **CORS is open to all origins** (`allow_origins=["*"]` in `api/main.py`). Restrict it to your front-end's address before deploying.
- **Do not expose the API to the internet** without a reverse proxy, rate limits and a review of the code paths above.

## Using it from Python

```python
import requests

base = "http://localhost:8000/api/v1"
token = requests.post(f"{base}/auth/login",
                      json={"email": "you@example.com", "password": "your-password"}).json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print(requests.get(f"{base}/models/", headers=headers).json())
```
