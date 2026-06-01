# ---------- Stage 1 : Build dependencies ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps required to compile numpy, pandas, shap, xgboost, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --prefix=/install -r requirements.txt


# ---------- Stage 2 : Final lightweight image ----------
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from generating .pyc files and force log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only the necessary project files
COPY api/ api/
COPY src/ src/
COPY models/best_xgboost_tuned.joblib models/best_xgboost_tuned.joblib
COPY api/main.py api/main.py
COPY params.yml .
COPY data/processed/data_processed.joblib data/processed/data_processed.joblib

# Expose FastAPI port
EXPOSE 8000
ENV PORT=8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
