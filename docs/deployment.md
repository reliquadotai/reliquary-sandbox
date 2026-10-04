# Deployment

## Host and trust boundary

Run the controller on a dedicated Linux Docker host. The controller has access to the host Docker socket, which grants control over that host. Tenant workloads run in separate gVisor containers; they receive no Docker socket, host mount, published port, or public network interface. Do not place the controller on a machine that stores unrelated production credentials.

The Compose controller runs as root to access the root-owned Docker socket. Its root filesystem is read-only, capabilities are dropped, and mutable service state lives in the `state` volume. These controller container settings do not reduce the authority granted by the Docker socket.

The guest image runs as UID 1000. The runtime supplies writable `/workspace` and `/tmp` mounts, a read-only root filesystem, and configured resource bounds. Guest images must contain Python 3 at `/usr/local/bin/python3` and `/bin/bash` and must support the configured guest UID. Build images before adding their immutable image references to the approved image list.

## Remote clients

Keep the controller bound to `127.0.0.1`. Put an existing HTTPS reverse proxy in front of port 8000, restrict administrative access to the host, and forward the `Authorization` header. Set `RELIQUARY_SANDBOX_PUBLIC_BASE_URL` to the HTTPS origin visible to clients; gateway URLs are derived from this origin. Do not add a URL path prefix.

The proxy must forward binary `application/connect+proto` request and response bodies and stream responses without buffering. Set proxy request limits to accommodate approved file transfers and idle/read timeouts longer than the maximum command duration. The controller does not trust proxy headers by default.

Give each tenant a distinct generated API credential in `RELIQUARY_SANDBOX_TOKENS`. Keep `.env` private and distribute only the relevant tenant credential. An API credential can manage all sandboxes owned by that tenant. Guest gateway credentials are scoped and expire; they are not tenant administrative keys. Rotating the configured tenant mapping requires a controller restart.

## Configuration

All controller settings use the `RELIQUARY_SANDBOX_` prefix. Lists and mappings are JSON values. See `.env.example` for Compose settings. When running directly, pass additional settings through the process environment.

| Setting | Default | Purpose |
| --- | --- | --- |
| `TOKENS` | Required | Tenant name to generated credential mapping |
| `PUBLIC_BASE_URL` | `http://127.0.0.1:8000` | Client-visible API and gateway origin |
| `STATE_DIR` | `data`; `/data` in Compose | Persistent metadata and checkpoint storage |
| `DOCKER_RUNTIME` | `runsc` | Guest container runtime |
| `DOCKER_HOST` | Local Docker socket | Controller-only Docker API connection |
| `REQUIRE_GVISOR` | `true` | Require the configured isolation runtime |
| `ALLOWED_IMAGES` | `["reliquary-sandbox-guest:0.1.0"]` | Guest image approval list |
| `DEFAULT_IMAGE` | `reliquary-sandbox-guest:0.1.0` | Default approved image |
| `DEFAULT_UID` | `1000` | Non-root guest UID |
| `PIDS_LIMIT` | `128` | Guest process limit |
| `MEMORY_LIMIT` | `1g` | Per-guest requested memory ceiling |
| `NANO_CPUS` | `1000000000` | Per-guest requested CPU ceiling |
| `WORKSPACE_SIZE_BYTES` | `5368709120` | Maximum writable workspace size |
| `TMP_SIZE_BYTES` | `67108864` | Guest temporary filesystem size |
| `MAX_FILE_BYTES` | `16777216` | Individual file transfer bound |
| `MAX_SNAPSHOT_BYTES` | `16777216` | Workspace snapshot bound |
| `MAX_SNAPSHOT_FILES` | `1000` | Workspace snapshot entry bound |
| `MAX_OUTPUT_BYTES` | `1048576` | Captured output bound |
| `MAX_SANDBOXES_PER_TENANT` | `16` | Concurrent tenant sandbox limit |
| `MAX_CPU_PER_TENANT` | `16` | Aggregate requested CPU limit |
| `MAX_MEMORY_GB_PER_TENANT` | `16` | Aggregate requested memory limit |
| `MAX_TIMEOUT_MINUTES` | `120` | Sandbox lifetime ceiling |
| `MAX_COMMAND_SECONDS` | `300` | Command duration ceiling |
| `GATEWAY_TOKEN_SECONDS` | `900` | Gateway credential lifetime |
| `MAX_CHECKPOINTS_PER_TENANT` | `64` | Stored checkpoint count limit |
| `REAPER_INTERVAL_SECONDS` | `5` | Expiry reconciliation interval |
| `RETENTION_DAYS` | `7` | Maximum terminal lifecycle history age |
| `MAX_TERMINAL_RECORDS_PER_TENANT` | `1000` | Maximum retained terminal lifecycle records |

CPU, memory, disk, and lifetime requests must fit the deployment's configured bounds. `/workspace` is a RAM-backed tmpfs charged to the container memory limit. Its requested size is an upper bound; requesting a 5 GiB workspace does not give a 1 GiB container additional memory or an independent disk.

## State and upgrades

Persist and back up the controller state volume. Checkpoints capture regular files inside `/workspace`, within the configured size and entry limits. They do not capture running processes, memory, installed system packages, or `/tmp`. A sandbox deletion removes the live guest; retained workspace checkpoints can be restored into another guest.

Creation idempotency is scoped to a tenant and its supplied key. Reusing a key with different parameters returns a conflict. Terminal sandbox history, including completed creation keys, is retained for up to seven days and the newest 1,000 terminal records per tenant. A key can create a new sandbox after its prior record is pruned. Checkpoints have their own count quota and remain until explicitly deleted. Resources awaiting runtime cleanup remain tracked and count toward active quotas.

Each state volume has a private runtime ownership identity. Keep that identity with the database when restoring a controller; separate state volumes can share a Docker host without reconciling each other's guests. Run one controller process per state volume. Concurrent commands or file operations on the same sandbox return a busy conflict; use separate sandboxes for parallel evaluations.

Finish file-writing work before creating a checkpoint. The archive copies workspace files without freezing background processes; it does not promise a consistent application snapshot while independent writers are active.

Stop accepting new work and allow active evaluations to finish before upgrading. Back up the state volume and private credentials, update the pinned dependencies and images, run the Linux integration suite, and then rebuild the controller. Monitor `/readyz` and container restart counts after rollout. `/health` checks controller responsiveness; `/readyz` also checks Docker, the approved images, runtime registration, and expiry reconciliation. Do not remove the state volume while checkpoints are required.

`docker compose down` stops the controller. Guests are managed by the controller and have their own lifetime limits; explicitly delete active sandboxes before shutting the controller down for an extended period.

## Direct process launch

```bash
uv sync --frozen --extra service
export RELIQUARY_SANDBOX_STATE_DIR=./data
# Supply the generated tenant mapping in RELIQUARY_SANDBOX_TOKENS.
uv run uvicorn reliquary_sandbox_service.app:create_app \
    --factory --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Installing the SDK alone does not install or provision a Docker host. The runtime bootstrap pins gVisor `20260928.0` and verifies the official archive's SHA-512 digest for Linux x86-64 or ARM64, including the runtime's companion binaries.
