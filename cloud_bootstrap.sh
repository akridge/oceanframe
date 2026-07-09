#!/usr/bin/env bash
set -euo pipefail

# Cloud host bootstrap for OceanFrame Web.
# Purpose:
# - Install Docker and prerequisites on Debian/Ubuntu hosts
# - Clone or update this repository
# - Start the Docker Compose stack
# - Install a systemd unit so the stack starts on reboot
#
# Common usage:
#   sudo bash cloud_bootstrap.sh
# Optional override:
#   sudo REPO_URL=https://github.com/akridge/oceanframe bash cloud_bootstrap.sh

# Deployment configuration (override via environment variables).
APP_NAME="${APP_NAME:-oceanframe-web}"
REPO_URL="${REPO_URL:-https://github.com/akridge/oceanframe}"
BRANCH="${BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-/opt/${APP_NAME}}"
SERVICE_NAME="${SERVICE_NAME:-${APP_NAME}.service}"

require_root() {
	if [[ "$(id -u)" -ne 0 ]]; then
		echo "Run this script as root: sudo bash cloud_bootstrap.sh" >&2
		exit 1
	fi
}

log() {
	echo
	echo "==> $1"
}

require_cmd() {
	command -v "$1" >/dev/null 2>&1 || {
		echo "Missing required command: $1" >&2
		exit 1
	}
}

clone_or_update_repo() {
	if [[ -d "${INSTALL_DIR}/.git" ]]; then
		git -C "${INSTALL_DIR}" fetch origin "${BRANCH}"
		git -C "${INSTALL_DIR}" checkout "${BRANCH}"
		git -C "${INSTALL_DIR}" reset --hard "origin/${BRANCH}"
	else
		git clone --branch "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
	fi
}

install_docker() {
	if ! command -v docker >/dev/null 2>&1; then
		log "Installing Docker and prerequisites"
		apt-get update
		apt-get install -y ca-certificates curl gnupg git docker.io docker-compose-plugin
	fi
	systemctl enable --now docker
}

write_systemd_unit() {
	local unit_path="/etc/systemd/system/${SERVICE_NAME}"
	cat > "${unit_path}" <<EOF
[Unit]
Description=OceanFrame Web compose stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${INSTALL_DIR}
ExecStart=/usr/bin/docker compose -f ${INSTALL_DIR}/docker-compose.yml up -d
ExecStop=/usr/bin/docker compose -f ${INSTALL_DIR}/docker-compose.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

	systemctl daemon-reload
	systemctl enable "${SERVICE_NAME}"
}

main() {
	require_root
	require_cmd apt-get
	require_cmd systemctl
	require_cmd git

	log "Preparing host"
	install_docker

	log "Cloning or updating repository"
	mkdir -p "$(dirname "${INSTALL_DIR}")"
	clone_or_update_repo

	log "Starting compose stack"
	cd "${INSTALL_DIR}"
	docker compose up -d --build

	log "Installing boot-time service"
	write_systemd_unit
	systemctl start "${SERVICE_NAME}"

	echo
	echo "OceanFrame Web is deployed."
	echo "Service   : ${SERVICE_NAME}"
	echo "Install   : ${INSTALL_DIR}"
	echo "Access    : http://<host-ip>/"
	echo "Restart   : systemctl restart ${SERVICE_NAME}"
	echo "Status    : systemctl status ${SERVICE_NAME}"
}

main "$@"