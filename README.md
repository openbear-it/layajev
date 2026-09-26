# LayaJev

LayaJev exposes [Laya](https://huggingface.co/convaiinnovations/laya) through the
TypeSafe Jev-compatible `POST /v1/systemone` API. It is designed for CPU-only
deployments and multi-architecture containers, including ARM64.

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt -e .
LAYA_CACHE_DIR="$HOME/.cache/layajev" \
  .venv/bin/uvicorn layajev.main:app --app-dir src --host 0.0.0.0 --port 8000
```

`requirements-cpu.txt` pins the CPU-only PyTorch wheel. Do not install Laya
directly from PyPI on a CPU-only host: its unconstrained PyTorch dependency can
select a much larger CUDA-enabled build.

Open `http://localhost:8000/` for the request console, or use:

```bash
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"document": "I was charged twice. Please refund me."},
    "questions": {
      "billing": {"type": "noul", "instructions": "Is this about billing?"}
    }
  }'
```

The first inference downloads a checkpoint into the persistent cache. Do not use
Uvicorn's `--reload` until the first request has finished: a reload stops the
worker and interrupts its active download. Laya may also download separate
English and multilingual checkpoints as requests require them. Subsequent calls
reuse completed files in the same cache directory.

By default the service uses CPU and retains at most one model in memory.
Configure it with:

- `LAYA_CACHE_DIR`: persistent checkpoint directory. Set it to
  `$HOME/.cache/layajev` locally or a mounted volume in containers.
- `LAYA_DEVICE`: Laya device, default `cpu`.
- `LAYA_MAX_LOADED`: resident checkpoint limit, default `1`.
- `LAYA_PRELOAD`: preload checkpoints at startup, default `0`.
- `API_KEY`: optional token; requests then require `Authorization: Bearer <token>`.
  `LAYA_API_KEY` remains supported as a compatibility alias for Laya deployments.

## Docker

```bash
docker build -t layajev .
docker run --rm -p 8000:8000 -v laya-cache:/models layajev
```

The container already sets its persistent `HF_HOME` cache to `/models`.

The published image is available at `ghcr.io/openbear-it/layajev` after a tag
or a push to `main`.