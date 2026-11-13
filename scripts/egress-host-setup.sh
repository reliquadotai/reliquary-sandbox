#!/bin/bash
# Controlled egress for online task boxes, and the host firewall for sidecar networks, on one
# gateway host (docs/egress.md). Run as root,
# after Docker and the gateway venv exist. Idempotent.
#   egress-host-setup.sh <venv> [snapshot-stamp, e.g. 20261101T000000Z]
# Makes: the internal bridge rq-egress (172.31.250.0/24, no inter-box traffic, IPv4 only),
# host firewall rules (boxes reach only the proxy ports on the bridge address), the proxy
# config /etc/reliquary-sandbox/egress.json and the systemd unit rq-egress-proxy.
# Prints the gateway settings to add (RELIQUARY_SANDBOX_EPISODE_EGRESS_*).
set -euo pipefail
VENV=$1; STAMP=${2:-$(date -u -d '-2 days' +%Y%m%dT000000Z)}
NET=rq-egress; BR=rq-egress0; SUBNET=172.31.250.0/24; GW=172.31.250.1; PORT=3128
E=/etc/reliquary-sandbox
[[ "$STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "bad snapshot stamp"; exit 2; }

if ! docker network inspect $NET >/dev/null 2>&1; then
  docker network create --driver bridge --internal --subnet $SUBNET --gateway $GW \
    -o com.docker.network.bridge.name=$BR -o com.docker.network.bridge.enable_icc=false $NET >/dev/null
fi
# Docker gives an internal bridge no address on some versions: the proxy needs one.
ip addr show dev $BR 2>/dev/null | grep -q "inet $GW/" || ip addr add $GW/24 dev $BR 2>/dev/null || true
ip link set $BR up || true

# Boxes -> host: only the proxy port. Boxes -> anywhere else: the bridge is internal (Docker
# drops forwarding); DOCKER-USER repeats it in case. Re-applied at boot by the unit below.
cat > /usr/local/sbin/rq-egress-firewall <<FW
#!/bin/sh
set -e
for chain in INPUT DOCKER-USER; do iptables -N RQ-EGRESS-\$chain 2>/dev/null || iptables -F RQ-EGRESS-\$chain; done
iptables -A RQ-EGRESS-INPUT -p tcp -d $GW --dport $PORT -j ACCEPT
iptables -A RQ-EGRESS-INPUT -j DROP
iptables -A RQ-EGRESS-DOCKER-USER -j DROP
iptables -C INPUT -i $BR -j RQ-EGRESS-INPUT 2>/dev/null || iptables -I INPUT 1 -i $BR -j RQ-EGRESS-INPUT
iptables -N DOCKER-USER 2>/dev/null || true
iptables -C DOCKER-USER -i $BR -j RQ-EGRESS-DOCKER-USER 2>/dev/null || iptables -I DOCKER-USER 1 -i $BR -j RQ-EGRESS-DOCKER-USER
ip6tables -C INPUT -i $BR -j DROP 2>/dev/null || ip6tables -I INPUT 1 -i $BR -j DROP
# Sidecar networks (rq-ep-*, internal, one per box with services) never reach the host.
iptables -C INPUT -i rq-ep+ -j DROP 2>/dev/null || iptables -I INPUT 1 -i rq-ep+ -j DROP
ip6tables -C INPUT -i rq-ep+ -j DROP 2>/dev/null || ip6tables -I INPUT 1 -i rq-ep+ -j DROP
FW
chmod 0755 /usr/local/sbin/rq-egress-firewall
/usr/local/sbin/rq-egress-firewall

install -d -m 0755 $E /var/log/rq-egress
cat > $E/egress.json <<JSON
{
  "listen": [{"host": "$GW", "port": $PORT, "profile": "packages"}],
  "sources": ["$SUBNET"],
  "log": "/var/log/rq-egress/egress.jsonl",
  "limits": {"connections_per_source": 32, "connects_per_minute": 600,
             "bytes_per_window": 8589934592, "window_s": 3600, "rate_bytes_per_s": 50000000,
             "max_connections": 2048, "total_rate_bytes_per_s": 400000000},
  "profiles": {
    "packages": {
      "ports": [80, 443],
      "hosts": ["pypi.org", "files.pythonhosted.org",
                "archive.ubuntu.com", "security.ubuntu.com", "ports.ubuntu.com", "snapshot.ubuntu.com",
                "deb.debian.org", "security.debian.org", "snapshot.debian.org",
                "dl-cdn.alpinelinux.org",
                "github.com", "codeload.github.com", "objects.githubusercontent.com",
                "release-assets.githubusercontent.com", "raw.githubusercontent.com",
                "registry.npmjs.org", "astral.sh", "releases.astral.sh",
                "static.rust-lang.org", "index.crates.io", "static.crates.io",
                "proxy.golang.org", "sum.golang.org",
                "repo.anaconda.com", "conda.anaconda.org"],
      "rewrites": [
        {"from": "http://archive.ubuntu.com/ubuntu/", "to": "https://snapshot.ubuntu.com/ubuntu/$STAMP/"},
        {"from": "http://security.ubuntu.com/ubuntu/", "to": "https://snapshot.ubuntu.com/ubuntu/$STAMP/"},
        {"from": "http://deb.debian.org/debian/", "to": "https://snapshot.debian.org/archive/debian/$STAMP/"},
        {"from": "http://deb.debian.org/debian-security/", "to": "https://snapshot.debian.org/archive/debian-security/$STAMP/"}
      ]
    }
  }
}
JSON
"$VENV/bin/python" -I -c "import json,sys; from reliquary_sandbox_service.egress_proxy import load_config; load_config(json.load(open(sys.argv[1])))" $E/egress.json

# The proxy may not reach the host's own or private networks (systemd cgroup filter; the
# proxy refuses them too): only its listener's subnet and the local resolver.
cat > /etc/systemd/system/rq-egress-proxy.service <<UNIT
[Unit]
Description=reliquary-sandbox egress proxy (online task boxes)
After=docker.service network-online.target
Wants=network-online.target

[Service]
ExecStartPre=+/usr/local/sbin/rq-egress-firewall
ExecStart=$VENV/bin/python -I -m reliquary_sandbox_service.egress_proxy $E/egress.json
DynamicUser=yes
LogsDirectory=rq-egress
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
NoNewPrivileges=yes
CapabilityBoundingSet=
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
IPAddressAllow=$SUBNET 127.0.0.53/32
IPAddressDeny=localhost link-local multicast 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 100.64.0.0/10 fc00::/7
MemoryMax=2G
TasksMax=512
LimitNOFILE=65536
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable -q rq-egress-proxy
systemctl restart rq-egress-proxy
sleep 1; systemctl is-active -q rq-egress-proxy

cat <<OUT
RELIQUARY_SANDBOX_EPISODE_EGRESS_NETWORK='$NET'
RELIQUARY_SANDBOX_EPISODE_EGRESS_PROFILES='{"packages":{"proxy":"http://$GW:$PORT","env":{"UV_EXCLUDE_NEWER":"${STAMP:0:4}-${STAMP:4:2}-${STAMP:6:2}T00:00:00Z","PIP_UPLOADED_PRIOR_TO":"${STAMP:0:4}-${STAMP:4:2}-${STAMP:6:2}T00:00:00Z"}}}'
# per env that may go online, e.g.:
RELIQUARY_SANDBOX_EPISODE_EGRESS_ENVS='{"<env>":["packages"]}'
OUT
