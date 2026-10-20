#!/usr/bin/env bash
# A second, isolated Docker daemon whose data-root is a loop-mounted xfs with
# project quotas, so episode disk-quota tests can run on a host whose Docker
# storage is not overlay2 on xfs. For test hosts only, never on production.
#
# It must never disturb the host's main daemon: own data-root (on the loop
# file), socket, exec-root and pidfile, its own containerd namespace, and its
# own private bridge `rqxfs0` with iptables, ip-forward and masquerading off.
#
# NEVER configure it with `"bridge": "none"`: on start, moby then deletes the
# host's `docker0` (removeDefaultBridgeInterface), which cuts the main daemon's
# container networking on the test host. `up` snapshots the host's
# links first and tears itself down if docker0 or any other pre-existing link
# changed (veth* links are ignored: the main daemon's containers churn them).
#
#   sudo scripts/test-xfs-docker.sh up      # create/mount the xfs file, start the daemon
#   eval "$(scripts/test-xfs-docker.sh env)"   # point docker and the live tests at it
#   sudo scripts/test-xfs-docker.sh down    # stop the daemon, unmount, delete rqxfs0 if up made it
#   sudo scripts/test-xfs-docker.sh purge   # down, then delete what up created
set -euo pipefail

REQUESTED=${XFS_DOCKER_ROOT:-/opt/reliquary-xfs}
case "$REQUESTED" in
  *..*) echo "XFS_DOCKER_ROOT must not contain '..'" >&2; exit 2 ;;
esac
ROOT=$(realpath -m -- "$REQUESTED")
case "$ROOT" in
  /opt/?*) ;;
  *) echo "XFS_DOCKER_ROOT must resolve under /opt (got $ROOT)" >&2; exit 2 ;;
esac
SIZE=${XFS_DOCKER_SIZE:-20G}
RUNSC=${RUNSC:-/usr/local/bin/runsc}
SOCK=/run/reliquary-xfs-docker.sock
PIDFILE=/run/reliquary-xfs-docker.pid
EXEC_ROOT=/run/reliquary-xfs-docker
NAMESPACE=reliquary-xfs
BRIDGE=rqxfs0
BRIDGE_ADDRESS=10.231.0.1/24
BRIDGE_MARKER=/run/reliquary-xfs-docker.bridge-created  # up created $BRIDGE: down may delete it

# The start time of our daemon, optionally matching a previously observed one.
# Keep checking the process after dockerd removes its pidfile during shutdown.
daemon_identity() {
  local pid=$1 stat started cmdline
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  cmdline=$(tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline") || return 1
  grep -qF -- "--config-file $ROOT/daemon.json " <<< "$cmdline" || return 1
  stat=$(cat "/proc/$pid/stat" 2>/dev/null) || return 1
  # Fields after the final ')' start at field 3; starttime is field 22.
  started=$(awk '{print $20}' <<< "${stat##*) }")
  case "$started" in ''|*[!0-9]*) return 1 ;; esac
  [ -z "${2:-}" ] || [ "$started" = "$2" ] || return 1
  echo "$started"
}

# The pid of OUR daemon, or nothing: the pidfile alone is not trusted.
ours() {
  local pid
  pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  daemon_identity "$pid" >/dev/null || return 1
  echo "$pid"
}

# Non-veth links as "name mac", sorted (states churn with the main daemon's containers).
links() { ip -br link | awk '$1 !~ /^veth/ { sub(/@.*/, "", $1); print $1, $3 }' | sort; }

# Mount release can lag behind process exit. Retry normal unmounts, and keep
# cleanup fatal if the owned mount is still busy after the bounded wait.
unmount_owned() {
  local target=$1
  for _ in $(seq 60); do
    mountpoint -q "$target" || return 0
    umount "$target" && return 0
    sleep 1
  done
  mountpoint -q "$target" || return 0
  echo "could not unmount $target" >&2
  return 1
}

down() {
  local pid identity
  if pid=$(ours) && identity=$(daemon_identity "$pid"); then
    if daemon_identity "$pid" "$identity" >/dev/null; then
      if ! kill "$pid"; then
        if daemon_identity "$pid" "$identity" >/dev/null; then return 1; fi
      fi
    fi
    for _ in $(seq 60); do daemon_identity "$pid" "$identity" >/dev/null || break; sleep 1; done
    if daemon_identity "$pid" "$identity" >/dev/null; then echo "the daemon did not stop" >&2; return 1; fi
  fi
  # What the daemon left mounted under its own exec-root (its null netns).
  { findmnt -rn -o TARGET | grep "^$EXEC_ROOT/" || true; } | sort -r | while read -r target; do
    unmount_owned "$target" || return 1
  done || return 1
  unmount_owned "$ROOT/mnt" || return 1
  if [ -f "$BRIDGE_MARKER" ]; then
    if ip -d link show "$BRIDGE" >/dev/null 2>&1; then ip link del "$BRIDGE"; fi
    rm -f "$BRIDGE_MARKER"
  fi
}

case "${1:-}" in
  up)
    command -v mkfs.xfs >/dev/null || { echo "install xfsprogs first" >&2; exit 1; }
    test -x "$RUNSC" || { echo "runsc not found at $RUNSC" >&2; exit 1; }
    ip link show docker0 >/dev/null 2>&1 || { echo "docker0 is missing before up: refusing" >&2; exit 1; }
    BEFORE=$(links | grep -v "^$BRIDGE " || true)
    if ip link show "$BRIDGE" >/dev/null 2>&1; then
      ip -d link show "$BRIDGE" | grep -q " bridge " || { echo "$BRIDGE exists and is not a bridge" >&2; exit 1; }
    else
      ip link add "$BRIDGE" type bridge
      : > "$BRIDGE_MARKER"
      ip addr add "$BRIDGE_ADDRESS" dev "$BRIDGE"
    fi
    ip link set "$BRIDGE" up
    mkdir -p "$ROOT/mnt"
    if [ ! -f "$ROOT/disk.img" ]; then
      truncate -s "$SIZE" "$ROOT/disk.img"
      mkfs.xfs -q "$ROOT/disk.img"
    fi
    mountpoint -q "$ROOT/mnt" || mount -o loop,pquota "$ROOT/disk.img" "$ROOT/mnt"
    cat > "$ROOT/daemon.json" <<JSON
{
  "data-root": "$ROOT/mnt/docker",
  "exec-root": "$EXEC_ROOT",
  "pidfile": "$PIDFILE",
  "hosts": ["unix://$SOCK"],
  "containerd": "/run/containerd/containerd.sock",
  "containerd-namespace": "$NAMESPACE",
  "containerd-plugins-namespace": "$NAMESPACE-plugins",
  "storage-driver": "overlay2",
  "features": {"containerd-snapshotter": false},
  "bridge": "$BRIDGE",
  "iptables": false,
  "ip6tables": false,
  "ip-forward": false,
  "ip-masq": false,
  "runtimes": {"runsc": {"path": "$RUNSC"}}
}
JSON
    if ours >/dev/null; then
      echo "already running"
    else
      nohup dockerd --config-file "$ROOT/daemon.json" >"$ROOT/dockerd.log" 2>&1 </dev/null &
      for _ in $(seq 60); do
        docker -H "unix://$SOCK" info >/dev/null 2>&1 && break
        sleep 1
      done
    fi
    sleep 2
    AFTER=$(links | grep -v "^$BRIDGE " || true)
    if ! ip link show docker0 >/dev/null 2>&1 || [ "$BEFORE" != "$AFTER" ]; then
      echo "host links changed while starting the daemon: tearing it down" >&2
      diff <(echo "$BEFORE") <(echo "$AFTER") >&2 || true
      down
      exit 1
    fi
    docker -H "unix://$SOCK" info --format '{{.Driver}} {{json .DriverStatus}} {{json .Runtimes}}'
    ;;
  down)
    down
    ;;
  purge)
    down
    rm -f "$ROOT/disk.img" "$ROOT/daemon.json" "$ROOT/dockerd.log"
    rmdir "$ROOT/mnt" "$ROOT" 2>/dev/null || true
    rm -rf "$EXEC_ROOT"
    # The daemon's containerd namespaces: drop the leases it kept, then the
    # namespaces themselves (empty once its containers are gone).
    if command -v ctr >/dev/null; then
      for ns in "$NAMESPACE" "$NAMESPACE-plugins"; do
        { ctr -n "$ns" leases ls -q 2>/dev/null || true; } | while read -r lease; do
          ctr -n "$ns" leases rm "$lease" >/dev/null
        done
        ctr namespaces remove "$ns" >/dev/null 2>&1 || true
      done
    fi
    ;;
  env)
    echo "export DOCKER_HOST=unix://$SOCK EPISODE_LIVE_XFS=1 EPISODE_LIVE_DOCKER_PIDFILE=$PIDFILE"
    ;;
  *)
    echo "usage: $0 up|down|purge|env" >&2
    exit 2
    ;;
esac
