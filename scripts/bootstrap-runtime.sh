#!/usr/bin/env bash
# Install a pinned gVisor distribution. Docker registration is an explicit option.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This runtime installer requires a Linux host." >&2
  exit 1
fi
if [[ "${EUID}" != "0" ]]; then
  echo "Run as root to install into /usr/local/bin." >&2
  exit 1
fi
if [[ "${1:-}" != "" && "${1}" != "--configure-docker" ]]; then
  echo "Usage: $0 [--configure-docker]" >&2
  exit 2
fi

release="20260928.0"
architecture="$(uname -m)"
case "${architecture}" in
  x86_64)
    digest="c8d3a9fd4d4c4f5b8ff213caa4517356be128d18659ec4cde37828fe797f61a9725a602a846c81a8ed19c057a996515d31c081eba343ed4613a89951ba32ed59"
    ;;
  aarch64)
    digest="926538a4f20056d44838f230297ecec9192db2562e2523a207295f706b76126725f2ff7b4e59d747147510c5714057eeec862a0f77d43bf625746592b5f51b00"
    ;;
  *)
    echo "Unsupported architecture: ${architecture}" >&2
    exit 1
    ;;
esac

for dependency in curl sha512sum tar bzip2; do
  command -v "${dependency}" >/dev/null || { echo "Missing ${dependency}" >&2; exit 1; }
done

runtime_temp="$(mktemp -d)"
trap 'rm -rf "${runtime_temp}"' EXIT
archive="${runtime_temp}/gvisor.tar.bz2"
curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
  --retry 3 --output "${archive}" \
  "https://storage.googleapis.com/gvisor/releases/release/${release}/${architecture}/gvisor.tar.bz2"
printf '%s  %s\n' "${digest}" "${archive}" | sha512sum --check --status
mkdir -p /usr/local/bin
tar --bzip2 --extract --file "${archive}" --directory /usr/local/bin --no-same-owner
/usr/local/bin/runsc --version

if [[ "${1:-}" == "--configure-docker" ]]; then
  command -v docker >/dev/null || { echo "Install Docker before registering runsc." >&2; exit 1; }
  command -v systemctl >/dev/null || { echo "Docker registration requires systemd." >&2; exit 1; }
  /usr/local/bin/runsc install
  systemctl reload docker
  docker info --format '{{json .Runtimes}}'
else
  echo "Installed. Register with Docker using: sudo runsc install && sudo systemctl reload docker"
fi
