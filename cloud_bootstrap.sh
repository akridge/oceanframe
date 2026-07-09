#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_VERSION="2026.07.09-1"

# Cloud host bootstrap for OceanFrame Web.
# Purpose:
# - Install Docker and prerequisites on Debian/Ubuntu hosts
# - Clone or update this repository
# - Start the Docker Compose stack
# - Install a systemd unit so the stack starts on reboot
#
# Common usage:
#   sudo bash cloud_bootstrap.sh
# Curl-and-run usage:
#   curl -fsSL https://raw.githubusercontent.com/akridge/oceanframe/main/cloud_bootstrap.sh | sudo bash
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

has_systemd() {
	command -v systemctl >/dev/null 2>&1 || return 1
	[[ "$(ps -p 1 -o comm= 2>/dev/null || true)" == "systemd" ]]
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

install_prerequisites() {
	local need_install=0

	if ! command -v git >/dev/null 2>&1; then
		need_install=1
	fi

	if ! command -v docker >/dev/null 2>&1; then
		need_install=1
	elif ! docker compose version >/dev/null 2>&1; then
		need_install=1
	fi

	if [[ "${need_install}" -eq 1 ]]; then
		log "Installing prerequisites (git, docker, compose plugin)"
		apt-get update
		apt-get install -y ca-certificates curl gnupg git docker.io docker-compose-plugin
	fi

	if has_systemd; then
		if ! systemctl enable --now docker; then
			echo "systemctl is present but not functional; falling back to non-systemd docker startup." >&2
			if command -v service >/dev/null 2>&1; then
				service docker start || true
			fi
		fi
	elif command -v service >/dev/null 2>&1; then
		service docker start || true
	fi

	if ! docker info >/dev/null 2>&1; then
		echo "Docker daemon is not reachable. Start Docker manually on this host and rerun bootstrap." >&2
		exit 1
	fi
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

	echo "OceanFrame bootstrap version: ${BOOTSTRAP_VERSION}"

	log "Preparing host"
	install_prerequisites

	log "Cloning or updating repository"
	mkdir -p "$(dirname "${INSTALL_DIR}")"
	clone_or_update_repo

	log "Starting compose stack"
	cd "${INSTALL_DIR}"
	docker compose up -d --build

	log "Installing boot-time service"
	if has_systemd; then
		write_systemd_unit
		systemctl start "${SERVICE_NAME}"
	else
		echo "systemd not detected; skipping system service installation."
		echo "Use this command after reboot/login to bring stack up:"
		echo "  cd ${INSTALL_DIR} && docker compose up -d"
	fi

	echo
	echo "OceanFrame Web is deployed."
	echo "Bootstrap : ${BOOTSTRAP_VERSION}"
	echo "Service   : ${SERVICE_NAME}"
	echo "Install   : ${INSTALL_DIR}"
	echo "Access    : http://<host-ip>/"
	if has_systemd; then
		echo "Restart   : systemctl restart ${SERVICE_NAME}"
		echo "Status    : systemctl status ${SERVICE_NAME}"
	else
		echo "Restart   : cd ${INSTALL_DIR} && docker compose up -d"
		echo "Status    : cd ${INSTALL_DIR} && docker compose ps"
	fi
}

main "$@"