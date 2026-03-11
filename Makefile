.PHONY: install test lint build-docker run-demo clean weaviate-up weaviate-down match-products

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m py_compile $(find src -name "*.py") && echo "Syntax OK"

# Full local demo using sample data
run-demo:
	python3 -m src.cli.main build-dataset
	python3 -m src.cli.main train
	python3 -m src.cli.main predict
	python3 -m src.cli.main recommend-order --policy balanced
	python3 -m src.cli.main backtest --policy balanced

# Weaviate vector DB (required before running match-products)
weaviate-up:
	docker compose up -d
	@echo "Waiting for Weaviate to be ready..."
	@sleep 10
	@echo "Weaviate ready at http://localhost:8080"

weaviate-down:
	docker compose down

# Run product matching (requires Weaviate running: make weaviate-up)
match-products:
	python3 -m src.cli.main match-products

build-docker:
	docker build -t auto-order-mvp .

# Run a demo inside Docker using local sample data
docker-demo:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts auto-order-mvp build-dataset
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts auto-order-mvp train
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts auto-order-mvp recommend-order --policy balanced

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet artifacts/*.pkl artifacts/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
