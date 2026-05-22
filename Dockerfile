# CS611 MLE Group Project — FastAPI inference service
# Serves real-time dengue outbreak risk predictions

FROM python:3.11-slim

WORKDIR /app

# System deps for geopandas
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.136.1 \
    uvicorn==0.47.0 \
    redis==5.0.1 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.49 \
    xgboost==3.2.0 \
    lightgbm==4.6.0 \
    mlflow==3.12.0 \
    pandas==2.3.3 \
    numpy \
    scikit-learn==1.8.0

# Copy inference app
COPY inference/realtime_inference.py .

# Data and model directories are mounted as volumes — not baked into image
# See docker-compose.yml volumes section

EXPOSE 8000

CMD ["uvicorn", "realtime_inference:app", "--host", "0.0.0.0", "--port", "8000"]
