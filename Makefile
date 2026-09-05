.DEFAULT_GOAL := help
.PHONY: help up down restart logs ps health topics clean install test lint format typecheck check

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---- Infrastructure ---------------------------------------------------------

up: ## Start the infrastructure stack and wait for healthy services
	docker compose up -d --wait

down: ## Stop the stack, keeping volumes
	docker compose down

restart: down up ## Recreate the stack

logs: ## Follow logs from every service
	docker compose logs -f

ps: ## Show service status
	docker compose ps

health: ## Report the health of each container
	docker compose ps --format 'table {{.Name}}\t{{.Status}}'

topics: ## List Kafka topics
	docker compose exec redpanda rpk topic list --brokers localhost:9092

clean: ## Stop the stack and delete all volumes (destroys local data)
	docker compose down -v

# ---- Python -----------------------------------------------------------------

install: ## Sync the uv environment with all extras
	uv sync --all-extras

test: ## Run the test suite
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .

typecheck: ## Type-check with mypy
	uv run mypy

check: lint typecheck test ## Run every quality gate
