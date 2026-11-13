# Controlled egress and sidecars for task boxes

Task boxes are offline by default (`network_mode=none`). Two opt-in extensions, both off unless
configured:

- **Egress**: a task declares `SandboxTask.network = TaskNetwork(profile, agent=False, grading=True)`
  (or a verifiers env sets `"network": {...}` in its `episode_env_options`). Its boxes, in the
  phases it names, join one Docker network and reach the internet only through the gateway's
  egress proxy for that profile.
- **Sidecars**: a task declares `SandboxTask.services` (`TaskService`: name, pinned image, command,
  env, memory, CPUs): the compose services other than the one the agent works in. Each runs in its
  own gVisor box on a private network with the task's box, which reaches it by name.

## Egress

```
box (runsc) --rq-egress0 (internal bridge, no inter-box traffic)--> 172.31.250.1:3128
   egress proxy (systemd, DynamicUser, cgroup IP filter) --> allowlisted hosts, ports 80/443
```

- The bridge is `internal` (no route out) with `enable_icc=false`; the host firewall accepts from
  it only TCP to the proxy port on the bridge address (no sshd, no gateway API, no Docker).
  The controller refuses to start when the network is not like this.
- The proxy (`reliquary_sandbox_service.egress_proxy`, standard library only) serves `CONNECT`
  and absolute-URI `GET`/`HEAD`. A host must match the profile's allowlist (exact names or
  `*.domain`, never addresses), the port its ports; the proxy resolves the name and connects only
  to a global unicast address it checked (no loopback, private, link-local/metadata, CGNAT).
  The unit's cgroup filter denies private ranges too.
- Pinning: plain-HTTP apt mirrors are rewritten to dated snapshots (`snapshot.ubuntu.com`,
  `snapshot.debian.org`); the profile's env sets `UV_EXCLUDE_NEWER` and `PIP_UPLOADED_PRIOR_TO`
  (pip >= 26). Other downloads (GitHub releases, npm) are recorded, not pinned: tasks should
  name exact versions.
- Abuse caps per box (source address): 32 connections, 600 connects/min, 8 GiB/h, 50 MB/s; the
  whole proxy: 2048 connections, 400 MB/s. Over a cap: 429 or slowed.
- Log: `/var/log/rq-egress/egress.jsonl`, one line per connection (source, host, port, URL for
  plain HTTP, verdict, bytes, seconds). The gateway logs `egress box <box> episode <id> address
  <ip>` at creation, which maps a line to its episode.

Host setup: `scripts/egress-host-setup.sh <venv> [snapshot stamp]` (idempotent) makes the
network, the firewall (`rq-egress-firewall`, re-applied by the unit), `/etc/reliquary-sandbox/egress.json`
and the `rq-egress-proxy` unit, and prints the settings:

| Setting | Meaning |
|---|---|
| `EPISODE_EGRESS_NETWORK` | the Docker network online boxes join (`rq-egress`) |
| `EPISODE_EGRESS_PROFILES` | `{profile: {"proxy": "http://172.31.250.1:3128", "env": {...}}}` |
| `EPISODE_EGRESS_ENVS` | `{env: [profiles]}`: which envs may go online (others are refused 422) |

The capacity report lists `egress` (env -> profiles) when set. A task's network is part of its
fingerprint. Measured under runsc on a Targon VM (2026-11-13): pip, apt (snapshot), uv installer,
npm, git clone over HTTPS work; direct internet, metadata, host ports, Docker's resolver and other
boxes are unreachable.

## Sidecars

For each box (agent's and each grading box) with services, the controller creates an internal
network `rq-ep-<hash>` (inter-container traffic on, no route out; the host firewall drops
`rq-ep+` input), starts each sidecar (runsc, task capabilities, `no-new-privileges`, 256 pids,
2 GiB disk quota, its memory and CPUs), then the box with the sidecars' addresses in its read-only
`/etc/hosts`. Sidecars find each other through Docker's resolver (their network alias). Removing
the box removes its sidecars and network. The task's `limits.memory_bytes` must cover the
sidecars plus 256 MiB; the box gets the rest.

Grading boxes get fresh sidecars: a test that checks state the agent left in a sidecar (a database
it filled) needs that state extracted by the env, or the pod variant (docs: Catalyst
`docs/design/2026-11-13-sandbox-network-compose.md`).
