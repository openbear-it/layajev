FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    USE_TF=0 \
    LAYA_DEVICE=cpu \
    LAYA_MAX_LOADED=1 \
    HF_HOME=/models

WORKDIR /app

# The PyTorch CPU index avoids downloading CUDA runtime packages on x86_64 and ARM64.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && python -m pip install laya==0.3.20 fastapi "uvicorn[standard]>=0.30,<1.0"

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models
USER appuser

VOLUME ["/models"]
EXPOSE 8000

CMD ["uvicorn", "layajev.main:app", "--host", "0.0.0.0", "--port", "8000"]