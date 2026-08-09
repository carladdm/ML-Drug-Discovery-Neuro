.PHONY: setup train docker-build docker-run clean lint test

PYTHON = python3.12

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	. .venv/bin/activate && python -m src.modeling.train

docker-build:
	docker build -t ml-drug-discovery-neuro:latest .

docker-run:
	docker run --rm -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data ml-drug-discovery-neuro:latest

clean:
	rm -rf __pycache__ .pytest_cache .venv
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete