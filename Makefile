.PHONY: setup db-up db-down db-migrate db-seed test lint run

setup:
	python -m venv venv
	venv/Scripts/python -m pip install -r requirements.txt || venv/bin/python -m pip install -r requirements.txt

db-up:
	docker-compose up -d postgres

db-down:
	docker-compose down

db-seed:
	@echo "Seeding database with demo data..."
	venv\Scripts\python.exe db\seed.py

test:
	@echo "Running tests and generating coverage report..."
	venv\Scripts\python.exe -m pytest --cov=src --cov=api --cov-report=term-missing tests/

lint:
	venv/Scripts/python -m ruff check src tests || venv/bin/python -m ruff check src tests
	venv/Scripts/python -m black --check src tests || venv/bin/python -m black --check src tests

run:
	@echo "Placeholder for Phase 9"
