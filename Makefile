# Convenience wrappers around docker compose.  Everything here is a one-liner
# you could type by hand; see `make help`.

COMPOSE ?= docker compose
SHELL   := /bin/bash

.DEFAULT_GOAL := help
.PHONY: help env build build-ml up up-ml dev down logs shell test test-live quickstart browse clean nuke

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1;36m%-12s\033[0m %s\n", $$1, $$2}'

env:  ## Create .env from the example, with your uid/gid filled in
	@test -f .env || cp .env.example .env
	@# Rewrite the lines if present, append them if the file predates them.
	@grep -q '^UID=' .env && sed -i.bak "s/^UID=.*/UID=$$(id -u)/" .env || echo "UID=$$(id -u)" >> .env
	@grep -q '^GID=' .env && sed -i.bak "s/^GID=.*/GID=$$(id -g)/" .env || echo "GID=$$(id -g)" >> .env
	@rm -f .env.bak
	@echo "Wrote .env with UID=$$(id -u) GID=$$(id -g)"

build: env  ## Build the core image (~450 MB)
	$(COMPOSE) build app

build-ml: env  ## Build the ML image (CLIP + YOLO + SAM 3, ~3 GB)
	$(COMPOSE) --profile ml build app-ml

up: env  ## Start the app -> http://localhost:8080/library
	$(COMPOSE) up -d --build app
	@echo "OceanFrame library: http://localhost:$${PORT:-8080}/library"

up-ml: env  ## Start the app with the model stack (own port, 8081)
	$(COMPOSE) --profile ml up -d --build app-ml
	@echo "OceanFrame library (ML): http://localhost:$${ML_PORT:-8081}/library"

dev: env  ## Live-reload against your working tree
	$(COMPOSE) --profile dev up --build dev

down:  ## Stop everything (keeps the catalog)
	$(COMPOSE) --profile ml --profile dev down

logs:  ## Tail the app log
	$(COMPOSE) logs -f app

shell:  ## Shell inside the running app container
	$(COMPOSE) exec app bash

test: env  ## Run the offline suite in the container
	$(COMPOSE) --profile test run --rm --build test

test-live: env  ## Run the suite against NOAA's public bucket (needs the ML image)
	$(COMPOSE) --profile live run --rm --build test-live

quickstart: env  ## Index ~2,800 real NOAA images into the app's volume
	$(COMPOSE) --profile quickstart run --rm --build quickstart

browse: quickstart up  ## Index the NOAA data, then start the app on it
	@echo "Open http://localhost:$${PORT:-8080}/library"

clean:  ## Stop and delete the catalog volume (source buckets are untouched)
	$(COMPOSE) --profile ml --profile dev down -v

nuke: clean  ## Also delete the built images
	-docker rmi oceanframe:core oceanframe:ml
