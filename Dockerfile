# Fraud alert review dashboard.
# Build:  docker build -t fraud-dashboard .
# Run:    docker run -p 8501:8501 fraud-dashboard   ->  http://localhost:8501
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Only what the dashboard reads: code, the exported model, the frozen threshold and the data bundle
COPY src/fraud/ src/fraud/
COPY app/ app/
COPY models/fraud_model.txt models/decision.json models/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
