#!/usr/bin/env bash
set -euo pipefail

# Local Docker convenience script.
# - Builds images if needed
# - Starts services in detached mode
# - Prints container status
docker compose up -d --build
docker compose ps