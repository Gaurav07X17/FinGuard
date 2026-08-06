# Use lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# System deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for faster rebuilds
COPY requirements-dev.txt requirements-add.txt /app/
RUN pip install --upgrade pip
RUN if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
RUN if [ -f requirements-add.txt ]; then pip install -r requirements-add.txt; fi

# Copy application
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit (use 0.0.0.0 so it's reachable inside k8s)
CMD ["streamlit", "run", "phase5/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
