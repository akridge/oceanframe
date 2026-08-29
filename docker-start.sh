#!/usr/bin/env bash
set -euo pipefail

# Local Docker convenience script.  `make up` does the same thing and also
# writes .env with your uid/gid; this exists for anyone without make.
#
# Profiles (see docker-compose.yml):
#   --profile ml          the app with CLIP / YOLO / SAM 3
#   --profile dev         live reload against the working tree
#   --profile test        the offline suite
#   --profile live        the suite against gs://nmfs_odp_pifsc
#   --profile quickstart  index ~2,800 real NOAA images

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
	cp .env.example .env
	# Compose cannot see the shell's $UID (bash marks it readonly and does not
	# export it), so it has to be written into .env.
	printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" >> .env
	echo "Created .env (UID=$(id -u) GID=$(id -g))"
fi

docker compose up -d --build
docker compose ps

PORT="$(grep -E '^PORT=' .env | cut -d= -f2)"
echo
echo "OceanFrame library: http://localhost:${PORT:-8080}/library"
