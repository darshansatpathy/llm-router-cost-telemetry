.PHONY: install check redis train calibrate replay dashboards

redis:
	docker run -d --name llmrouter-redis -p 16379:6379 redis:8.2

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[embed,dev]"

check:
	ruff check .
	ruff format --check .
	mypy
	pytest --cov --cov-report=term-missing

train:
	llmrouter train

calibrate:
	llmrouter calibrate-cache

replay:
	llmrouter replay --config gateway.yaml --redis redis://127.0.0.1:16379/0 --out docs/replay.md

dashboards:
	python deploy/grafana/build.py
