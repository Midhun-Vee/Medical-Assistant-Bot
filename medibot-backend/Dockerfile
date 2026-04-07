FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install uv
RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run","--active", "uvicorn", "main:app", "--port", "8000", "--reload"]