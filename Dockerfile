FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Make data directories available (they may be empty in the image)
RUN mkdir -p data/raw data/processed data/samples artifacts

ENV PYTHONPATH=/app

# Default command shows help; override with specific CLI commands
ENTRYPOINT ["python", "-m", "src.cli.main"]
CMD ["--help"]
