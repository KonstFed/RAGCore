FROM python:3.12-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv pip install --system .

COPY src ./src
COPY app.py ./
COPY configs ./configs

EXPOSE 8501

CMD ["python", "app.py"]
