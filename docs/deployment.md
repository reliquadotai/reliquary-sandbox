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

## Episode hosts

An episode host runs the signed episode gateway (`EPISODES_ENABLED=true`). A miner opens an episode with a validator-signed session token, sends tool calls, and finishes. The machine signs every call's output and the final record with its Ed25519 key and chains the records by hash, so a validator can verify the transcript without re-executing anything. The sandbox knows nothing of payment, TOPLOC or the subnet; environments neither isolate nor sign. The episode routes are added only when episodes are enabled. Dedicate the host to episodes: the controller still requires tenant credentials, and they remain usable for administration.

Settings below use the `RELIQUARY_SANDBOX_` prefix, like every other controller setting. The sections that follow this one (runsc version check, readiness, helper busybox, measured behaviour, testing) give the details.

### Host requirements

- Linux with Docker Engine on a Unix socket, and the pinned gVisor runtime: `sudo ./scripts/bootstrap-runtime.sh --configure-docker` installs runsc `20260928.0` (SHA-512 checked) with `--overlay2=none`. Keep `REQUIRE_GVISOR=true`. The controller compares `runsc --version` with `EPISODE_RUNSC_VERSION` and refuses `--ignore-cgroups`. Quoted episode disks require the explicit `--overlay2=none` runtime argument with no conflicting overlay flags, so writes use Docker's quota-controlled layer. Qualify this configuration on a new or drained host before rollout.
- Docker's data-root on xfs mounted with `pquota`, with the `overlay2` storage driver and the containerd image store disabled. The controller caps each box's writable root with `--storage-opt size=<disk budget>`. Preflight refuses to enable episodes unless `docker info` reports `overlay2` on xfs and a probe container created with a size option is accepted. `EPISODE_ALLOW_UNQUOTED_DISK=true` skips the check and the quota; use it for tests only.
- A static busybox, at least 1.36, pinned by SHA-256 (`EPISODE_HELPER_BUSYBOX`, `EPISODE_HELPER_BUSYBOX_SHA256`). See "Episode hosts: the helper busybox".
- The controller must see the Docker pidfile (`EPISODE_DOCKER_PIDFILE`, default `/var/run/docker.pid`) and `/proc`, so it can tell a daemon restart from an agent's action. A controller in a container needs both mounted read-only, and the host's `runsc` at the path Docker reports.
- No host RAM overcommit: the boxes' memory limits plus the controller must fit in host RAM. A box killed for memory is charged to the agent.
- Task images pulled by digest before the controller starts (`docker pull repo@sha256:<digest>`) and listed in `EPISODE_IMAGES`. The controller never pulls. Images need `/bin/bash`; plumbing runs from the helper busybox, so they need no Python, `sh` or coreutils of their own. They run as the user and workdir the env declares.
- KVM is optional. It enables a faster gVisor platform; without it gVisor uses systrap.

`/etc/docker/daemon.json` on a host whose `/var/lib/docker` is an xfs volume mounted with `pquota`:

```json
{
  "storage-driver": "overlay2",
  "features": {"containerd-snapshotter": false},
  "runtimes": {"runsc": {"path": "/usr/local/bin/runsc", "runtimeArgs": ["--overlay2=none"]}}
}
```

### Machine and validator keys

Generate the machine key on the host. Only the controller reads it:

```bash
uv run python -m reliquary_sandbox.attest.signing generate /etc/reliquary-sandbox/machine.pem
```

The command refuses to overwrite an existing file, writes a mode-0600 PKCS#8 PEM, and prints the base64 public key. Register that public key, its key id and its validity window in the validator's machine directory. Then set `MACHINE_KEY_FILE` (mounted read-only into the controller), `MACHINE_KEY_ID` (the registered key id) and `MACHINE_ID` (the machine's directory id).

The controller refuses a key file that is a symlink, is readable by group or others, or is not owned by the controller's user. One machine has one gateway and one signing key, and every episode uses a single key id from its record 0 to its final record. To rotate, generate a new key, register it under a new key id valid from a start time, switch `MACHINE_KEY_ID` and `MACHINE_KEY_FILE`, and restart the controller. Rotate with no active episode (`GET /capacity` reports `active: 0`): an episode whose record 0 was signed with another key id is never signed again (a final under the new key would not verify). At recovery, or at its next call, finish or close, it is closed locally as `aborted_unsigned`, without a final record, its box removed and its slot freed, and the controller logs an error. The miner gets `409` with `final: null`. If a key is compromised, the validator should set its `valid_until` to the earliest time compromise is suspected. `VALIDATOR_PUBLIC_KEYS` maps validator key ids to the base64 Ed25519 public keys that sign session tokens.

Episode keys are not random: the key of an episode is `HMAC-SHA256(secret, token_sha256)`, where the secret is 32 random bytes the controller creates once in `<state_dir>/episode-secret` (mode 0600). A miner whose open response was lost sends the same token again and gets the same key back. Protect the file like the machine key: whoever reads it can derive the key of any episode whose token they hold. Deleting it changes every live episode's key.

### Environment packages

`EPISODE_ENVS` maps env names (the token's `env` claim) to `module:callable` entry points. Each entry point is called with `(split, index)` and returns a `reliquary_sandbox.episode_task.SandboxTask`. Install env packages into the controller's environment from a verified wheelhouse (procedure in "Episode hosts: env packages and task images"); with Compose, build a controller image derived from this one. Record 0, the task fingerprint and the capacity report carry each package's identity `distribution==version+g<digest>`: the digest is the first 16 hex of a sha256 over the distribution's installed `.py` files (sorted by path, each hashed as `path NUL bytes`), computed once when the registry loads. A grading file edited in place therefore changes the identity. Validators allow-list the `env_package` values they accept; a package that is not a wheel install (editable) is hashed from its imported package directory instead, and a module outside any distribution reports `<module>==unpackaged`. Validators compute the values to allow-list with `reliquary_sandbox_service.episodes.env_package_of(module_or_name)` (a module, a module name or a `module:callable` entry point), the function the registry uses. The task's image must equal the token's image and be listed in `EPISODE_IMAGES`.

An `EPISODE_ENVS` value may instead be `verifiers:<taskset-id>==<version>`: a standard verifiers v1 Taskset served through the generic bridge (`reliquary_sandbox_verifiers`), with no Reliquary code in the env. The controller imports the taskset at start, as verifiers does (`<taskset-id>` with `-` as `_`), but only after the checks pass, which read metadata without executing the module: it refuses to start unless the installed distribution's version is exactly `<version>` (a module outside any distribution is refused unless `EPISODE_ALLOW_UNPACKAGED_VERIFIERS=true`, tests only; its version is then `unpackaged`), and reads the package's image lock `sandbox-images.lock.json` (`{"version": 1, "images": {"<TaskData.image>": "<repo>@sha256:<digest>"}}`): a task whose image the lock does not pin is refused. `EPISODE_ENV_OPTIONS` (JSON, per env name, verifiers entries only) sets `tools` (default `["bash", "edit"]`), `splits` (served split name to taskset config overrides; default `{"rl": {"split": "rl"}}`; a split serving a taskset split whose name contains `eval` must itself be named with `eval`, and only such a split may be, so held-out tasks are never served under a training name: a gateway serves them only when its operator lists one), `defaults` (`BridgeDefaults`: workdir, pids, per-call timeout, max calls, wall, prepare and grading timeouts, artifact bytes) and `publish_state`. The capacity report publishes, per bridged env, `env_options_sha256`: the sha256 of the canonical JSON of its options in effect (`VerifiersEnvOptions.sha256`, defaults filled in, tools sorted); a validator that builds the same env with its own copy of the options compares it with `options_sha256(...)` and places no session on a machine that differs. A bridged env's identity is `bridged_env_package_of(module)` = `env_package_of(module)` + `.vfb<bridge version>`, so a change to how the bridge grades changes what validators allow-list; its record 0 `setup_output_sha256` is the sha256 of `{"bridge":<version>,"task_hash":"<Task.hash>"}` (compact JSON, sorted keys), which a validator recomputes from the task it loads. The task's content hash is also in the task fingerprint. `module:callable` entries stay supported for test and tool envs; the SWE and terminal envs are served as verifiers entries.

### Episode settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `EPISODES_ENABLED` | `false` | Serve the episode routes |
| `MACHINE_ID` | Required | Directory id; session tokens name it |
| `MACHINE_KEY_FILE` | Required | Ed25519 PKCS#8 PEM, mode 0600 |
| `MACHINE_KEY_ID` | Required | Key id registered in the directory |
| `VALIDATOR_PUBLIC_KEYS` | Required | Validator key id to base64 public key |
| `EPISODE_IMAGES` | Required | Approved task images, `repo@sha256:...` |
| `EPISODE_ENVS` | Required | Env name to `module:callable` or `verifiers:<taskset-id>==<version>` |
| `EPISODE_ENV_OPTIONS` | `{}` | Per verifiers env: `tools`, `splits`, `defaults`, `publish_state` (JSON) |
| `EPISODE_ALLOW_UNPACKAGED_VERIFIERS` | `false` | Serve a verifiers env that is not an installed distribution (tests only) |
| `EPISODE_HELPER_BUSYBOX` | Required | Path of the static busybox |
| `EPISODE_HELPER_BUSYBOX_SHA256` | Required | Its SHA-256, 64 hex digits |
| `EPISODE_RUNSC_VERSION` | `release-20260928.0` | Version `runsc --version` must report |
| `EPISODE_DOCKER_PIDFILE` | `/var/run/docker.pid` | Docker daemon pidfile |
| `EPISODE_FAULT_SETTLE_S` | `1.0` | Settle time before a failure is charged to infrastructure |
| `EPISODE_HOST_CHECK_SECONDS` | `30` | Interval of the host checks behind `/readyz` |
| `EPISODE_CAPACITY` | `4` | Concurrent episodes; beyond it, `503` with `Retry-After` |
| `EPISODE_NANO_CPUS` | `2000000000` | CPUs per task box |
| `EPISODE_CPU_MILLIS_TOTAL` | Unset | CPU (thousandths) all live episodes' agent boxes may reserve; with `EPISODE_MEMORY_BYTES_TOTAL` ("Episode hosts: readiness and capacity") |
| `EPISODE_MEMORY_BYTES_TOTAL` | Unset | Memory all live episodes' agent boxes may reserve; both or neither |
| `EPISODE_CPU_OVERCOMMIT` | `1.0` | Factor (1.0 to 4.0) on `EPISODE_CPU_MILLIS_TOTAL`, fixed at qualification |
| `EPISODE_GRADING_CPU_MILLIS_TOTAL` | Unset | CPU (thousandths) of the grading pool; required with the totals above |
| `EPISODE_GRADING_MEMORY_BYTES_TOTAL` | Unset | Memory of the grading pool; both or neither |
| `EPISODE_PRESENT_WINDOW_S` | Unset | A token presented later than its `issued_at` plus this is refused `401`, unsigned and unspent. RL machines: `120`, the validator's `PRESENT_WINDOW_S`; corpus machines: unset |
| `EPISODE_MAX_MEMORY_BYTES` | `8589934592` | Largest memory budget accepted |
| `EPISODE_MAX_PIDS` | `1024` | Largest pids budget accepted |
| `EPISODE_MAX_DISK_BYTES` | `10737418240` | Largest disk budget accepted |
| `EPISODE_MAX_WALL_S` | `14400` | Largest wall budget accepted |
| `EPISODE_MAX_CALLS` | `512` | Largest call budget accepted |
| `EPISODE_MAX_CPU_S` | `3600` | Largest CPU budget accepted |
| `EPISODE_MAX_CALL_TIMEOUT_S` | `600` | Largest per-call timeout accepted |
| `EPISODE_OUTPUT_CAP_BYTES` | `65536` | Tool output kept per call; beyond it, cut and flagged `truncated` |
| `EPISODE_EDIT_MAX_BYTES` | `4194304` | Largest file the `edit` tool rewrites |
| `EPISODE_GRADER_OUTPUT_BYTES` | `33554432` | Output limit of one grading step; beyond it the state is graded 0.0 |
| `EPISODE_GRADING_ATTEMPTS` | `2` | Pristine boxes tried before the episode ends `aborted` |
| `EPISODE_GRADING_CREATE_TIMEOUT_S` | `90` | Bound on a grading box's creation and first CPU reading (at most 120); past it the attempt fails on our side |
| `EPISODE_MAX_TRANSCRIPT_BYTES` | `8388608` | At most 8 MiB (Catalyst's wire cap). Bytes of the episode's transcript as validators receive it: compact UTF-8 JSON `{"token","records"}`, `ensure_ascii=False`. Every call is recorded; once what is left could not hold one more worst-case call record and the final (`transcript_reserve_bytes(EPISODE_OUTPUT_CAP_BYTES)`, about 0.6 MiB at the default output cap), the episode ends `budget_exhausted` (`transcript_bytes`), so the signed transcript, final included, never passes the cap. The cap must exceed that reserve by 128 KiB (lower `EPISODE_OUTPUT_CAP_BYTES` otherwise) |
| `EPISODE_RETENTION_DAYS` | `7` | History kept after a token expires; also the longest token validity accepted |
| `EPISODE_CPU_SOURCE` | `docker` | `docker` (one-shot stats) or `cgroupfs` |
| `EPISODE_CGROUP_ROOT` | `/sys/fs/cgroup` | Host cgroup v2 root for `cgroupfs` |
| `EPISODE_ALLOW_UNQUOTED_DISK` | `false` | Test hosts only: no xfs quota |
| `SCORING_UID_BASE` | `200000` | First uid (and gid) of the verifiers bridge's scoring slots; the range must be unused on the host ("The scoring users") |
| `SCORING_SLOTS` | `64` | Scoring slots (uids `SCORING_UID_BASE` + 0..n-1); at least `EPISODE_CAPACITY` |
| `SCORING_TMP_ROOT` | Unset (`/scoring` with `compose.episodes.yaml`) | Where scoring TMPDIRs are made; unset: the controller's temporary directory |
| `SCORING_WRAPPER` | `/usr/local/libexec/reliquary-scoring-python` | Executable that starts the scoring forkserver with an empty environment |
| `HEARTBEAT_URL` | Unset | Validator endpoint receiving signed capacity reports |
| `HEARTBEAT_INTERVAL_SECONDS` | `5` | Push interval |

Startup refuses an enabled configuration without a machine id, key file and key id, a validator key, digest-pinned images, env entry points and the pinned busybox. The shipped `compose.yaml` passes none of these variables: add the ones you set to the controller's `environment:` block and mount the key file and busybox read-only. Compose passes only the variables it lists. Episode hosts that serve `verifiers:` entries also apply `compose.episodes.yaml` on top of it (`docker compose -f compose.yaml -f compose.episodes.yaml up -d`, Docker Compose 2.24.4 or later): it adds the capabilities, limits, sysctl and tmpfs mounts the scoring processes need, and sets `SCORING_TMP_ROOT=/scoring` ("The scoring users" below). Tenant-API and corpus-executor hosts use `compose.yaml` alone and keep their previous settings.

### Endpoints

Request and response shapes are in [compatibility: signed episodes](compatibility.md#signed-episodes).

- `POST /episodes` is authenticated by the session token in its body. Every other episode route takes `Authorization: Bearer <episode_key>`, the key returned at open. Only its hash is stored, and it never appears in a URL.
- One call's arguments may take at most 122880 canonical bytes (120 KiB, below one argv element's limit), answered `413` beyond. This is fixed, not a setting: it is `reliquary_sandbox.observation.MAX_ARGUMENT_BYTES`, which miners and validators also use to refuse oversized calls locally.
- Request bodies are bounded before parsing: 16 KiB for a token, `6 * 122880 + 64 KiB` for a call, 1 KiB for everything else, answered `413`.
- A task may declare limits (`SandboxTask.limits`): a session token whose memory, disk or pids budget is below them is refused `422`, signed and spent (see "Signed open refusals" below); the task's CPUs size its boxes, capped by `EPISODE_NANO_CPUS`.
- `401` answers an invalid token, an expired or not-yet-valid one (30 s of validator clock skew is tolerated), an unknown episode and a wrong key alike. `403` is a token for another machine. `422` is a token the machine cannot serve: budgets over its caps, a task image it does not approve or whose grading cannot fit the verification window, a bad session id, validity over the retention, or malformed call arguments.
- `POST /episodes` is idempotent per token. Sending the same token again while its episode runs (or grades) returns `200` with record 0 and the same key, even on a full machine; while it is still opening, `409` busy (retry); once it is closed, `409` with the final. A different token for the same session id gets a plain `409`.
- `409` is a token already used, a call or finish while another request holds the episode (busy), a call out of order, a `(turn, k)` resent with different content, a transcript already being read, or an image missing on the machine. `409` with `{"detail": {"message", "final"}}` also answers any request on a closed episode. A `finish` that finds the deadline passed closes the episode `expired` and answers `409` with that final; later `finish` calls return it with `200`. When the reaper closed the episode first, the first `finish` already gets `200`. `final` is `null` for an episode closed without a final record (a failed open, or `aborted_unsigned` after a key change).
- `GET /episodes/{id}` builds the transcript from the stored records on its own two-thread pool, one read at a time per episode, holding the state store only page by page, so a long transcript never stalls other episodes.
- `503` with `Retry-After: 5` is a full machine (the token is spent and the refusal signed: below), a controller still recovering, or an unavailable capacity report. `503` without `Retry-After` and with the message "token spent" is a failed open: no record 0 exists, the token is burnt and the validator must issue a new one. `503` with `{"detail": {"message", "final"}}` is a sandbox failure during a call: the signed `aborted` final is attached.
- **Signed open refusals.** A refusal of a fresh token for a reason that is not the miner's (the box could not be prepared; no free slot or resources; claims this machine cannot serve: an unresolvable task, an image that is not the task's, not approved or not present, a grading timeout, budgets below the task's needs or above the machine's limits, a validity above the retention, a box too small, a reservation that never fits) spends the token and carries the header `X-Reliquary-Open-Refusal`: a receipt signed with the machine key under the open-refusal domain (`reliquary-sandbox/open-refusal/v1`), bound to the session id and the token's sha256, with its reason (`open_failed`, `no_free_slot`, `no_free_resources`, `task_refused`). The token is spent with it (a closed row under its session id, final status `refused`, pruned like any closed episode once the token expired): neither the token nor its session id ever opens later, so a receipt cannot be kept while a box runs after all. Sending the same token again answers `409` with the same receipt (nothing new is signed), so a lost response loses nothing. Miners forward it when they report the failed open (`EpisodeError.open_refusal` in the client); validators verify it with `reliquary_sandbox.attest.open_refusal_valid` and the machine's directory key, and then do not count the session against the miner. An invalid, expired, foreign, reused or busy token, and an invalid session id, carry none. A token presented too close to its expiry to prepare a box (its `expires_at` within the task's preparation timeout plus 60 s of now) is refused `401` unsigned and unspent, and so is a token presented after `EPISODE_PRESENT_WINDOW_S`; both checks come before any signed refusal, so holding a token until its last seconds never buys a receipt. A preparation that outlasts the token is not signed either. Validators must therefore issue tokens valid for longer than the task's preparation timeout plus 60 s and transit.
- `GET /capacity` is unauthenticated. It serves a signed report (cached for 1 s) of free slots, images, env packages, tool, runsc and busybox versions, and the machine's caps (`caps`: `max_calls`, `max_cpu_s`, `max_call_timeout_s`, `max_wall_s`, `max_memory_bytes`, `max_pids`, `max_disk_bytes`, `max_argument_bytes`, `output_cap_bytes`, `max_token_validity_s`, `max_transcript_bytes`; `heartbeat_valid` refuses a report missing one), signed under its own domain. With `HEARTBEAT_URL` set the same report is pushed every `HEARTBEAT_INTERVAL_SECONDS`, backing off after failures. Validators take a machine's address from their directory, never from a report.

### How an episode ends

Every episode ends with a signed final record whose status says whose doing it was:

| Status | Cause | Fault |
| --- | --- | --- |
| `graded` | `finish` ran extract and grading. The reward is in `[0, 1]`; a 0.0 can carry a `_grader` fact such as `state_too_large` or `grading_timeout` | Agent's outcome, paid by reward |
| `expired` | The wall deadline passed before `finish`, found by a call, a finish or the reaper | Agent's, no reward |
| `budget_exhausted` | `max_calls` or `cpu_s` ran out, or the transcript came within one worst-case call and final of `EPISODE_MAX_TRANSCRIPT_BYTES` (reason `transcript_bytes`) | Agent's, no reward |
| `box_failed` | The agent's box became unusable through what the agent did: it killed its processes, exhausted its pids, was killed for memory, deleted its workdir | Agent's, not paid, not voided |
| `aborted` | Our side failed: a Docker or sandbox failure, a restart that interrupted a call or grading, a lost box, an env calling a runtime member the sandbox does not offer, an env hook raising `EnvInfraError` (the env reporting a failure of its own, e.g. a pristine box that is not what its corpus says), an env that declares the task differently after a restart (reason `task_changed`) | Ours, void; the task can be run again |

An episode whose machine key changed under it ends `aborted_unsigned`: a local status with no final record (see key rotation above).

The fault rule: a failure is an `InfraFault` (ours, `aborted`) only when Docker raised, the box failed before it started, the daemon restarted after the box started, or Docker reported an error on it. Once the box has started, everything else (a helper that fails or times out, an output lost while the box still runs, an out-of-memory kill, PID 1 exiting) is a `BoxFault` (the agent's: `box_failed` on the agent's box, graded 0.0 with `grading_box_failed` in the grading box). A box that still reads `running` after a failure is re-checked after `EPISODE_FAULT_SETTLE_S` before it is charged to infrastructure. A call during which the box is gone ends the episode at that call, which is not recorded. The validator should cap `aborted` episodes per hotkey, because an agent that found a way to force one would get a free re-roll.

### Limits, timeouts and accounting

- Every task box runs under runsc with `--network none`, all capabilities dropped except `CHOWN`, `DAC_OVERRIDE`, `FOWNER`, `FSETID`, `KILL`, `SETGID`, `SETUID`, `SETPCAP`, `NET_BIND_SERVICE` and `SYS_CHROOT`, `no-new-privileges`, swap equal to memory, an init process, no container log, and hostname `sandbox`. `/etc/hosts`, `/etc/hostname` and `/etc/resolv.conf` are controller-owned and read-only. Its pids, memory and disk come from the session budgets, which the `EPISODE_MAX_*` settings bound.
- The agent's wall time starts when the task's `prepare` ends: the deadline is the earlier of that time plus `wall_s` and the token's `expires_at`. A preparation that outlasts the token fails the open. The box outlives the deadline by at least 10 minutes (longer for a task with a longer grading timeout), then stops on its own (its PID 1 is a `sleep`); the episode's close or the reaper removes it.
- When a `bash` call exceeds its timeout, only that call's process group is killed. The call returns `timed_out` with the output printed so far; the box and processes left by earlier calls keep running. The call's timeout is the smaller of the session's per-call timeout and the time left to the deadline. The group id is stored on the box's `/dev/shm`, outside the disk quota, so a full disk does not break timeouts. If it cannot be stored, the runner refuses to run the command and exits 125.
- A call's `cpu_ms` is the change in the box cgroup's CPU usage since the previous charge, read from Docker one-shot stats. When the controller runs on the host, `EPISODE_CPU_SOURCE=cgroupfs` reads `cpu.stat` under `EPISODE_CGROUP_ROOT` instead. The figure covers every process in the box, including gVisor's Sentry and gofer, so background jobs left by earlier calls are charged to the next call. CPU burnt after the last call is charged at `finish`. The `cpu_s` budget counts calls only; the final record's `cpu_total_ms` adds the grading box's CPU. Preparation is not counted.
- At `finish`, before extract, every process the agent left in its box is killed (as root, with the helper busybox; PID 1 and the entrypoint stay), so the extracted state is final and nothing tampers with extraction.
- A grading box must be created, and its CPU read once, within `EPISODE_GRADING_CREATE_TIMEOUT_S`; otherwise that attempt fails on our side (retried, then `aborted`). The bound leaves each grading step its full timeout inside the verification window, so a step cut by the window always follows our own slowness.
- Files move through `docker exec` from the helper busybox, preserving the box's user and file-operation checks. Quoted disks write directly to Docker's quota-controlled layer with `--overlay2=none`.

### Restarts

Episodes are stored in the controller's SQLite state, and recovery finishes before the routes serve. At startup the controller:

- keeps running episodes whose boxes survived, restoring each box's user and CPU reading before any call;
- fails any episode that was still opening (no record 0, token burnt, slot freed);
- closes `aborted_unsigned`, without signing, any episode whose record 0 carries another machine key id;
- closes with a signed `aborted` any episode that was mid-call or mid-grading, or whose box is gone or stopped (a host restart cannot be told from the agent stopping its box), and with `expired` any episode past its deadline;
- removes boxes that no active episode owns.

A call resent after a restart returns the stored record and does not run again. After a restart the task is resolved again and must be exactly the one the episode opened (same env package version, image, users, workdirs, environment, tools, limits and timeouts, compared by a fingerprint stored at open); otherwise the episode ends `aborted` with reason `task_changed` at its next call or finish. Upgrade env packages with no live episode. A session token opens one episode: sending it again returns that episode, and a failed open (after the controller accepted it) still uses it up. The reaper also settles an open that was left without its opener past its preparation time as a failed open. Closed episodes are pruned `EPISODE_RETENTION_DAYS` after closing, and only once the token has expired.

### For environment authors

An env exposes `sandbox_task(split, index)` returning a frozen `SandboxTask`: `image` (digest-pinned, equal to the token's), `workdir`, `tools` (a subset of `bash`, `edit`), `prepare`, `extract`, `grade`, optional `env`, `user`, `max_state_bytes` (32 MiB), `prepare_timeout_s` and `grading_timeout_s` (600 s each), `grading_workdir` and `grading_user` (default: the agent's), and `publish_state`. The hooks receive a `TaskRuntime`, the subset of the verifiers runtime that existing graders use.

- `prepare` runs in the agent's box before record 0 and returns text, hashed into record 0. If it raises, the open fails and the token is burnt.
- `extract` runs in the agent's box at `finish` and returns the state bytes. `grade` runs in a pristine box from the same image and never sees the agent's box.
- Failure contract: an infrastructure fault is `aborted`; a fault of the agent's box during `extract` is `box_failed`; everything else is graded 0.0 with a `_grader` fact. That includes an exception from your own `extract` or `grade` (`extract_failed`, `grading_failed`, logged and counted per env, so an env bug zeroes honest miners visibly rather than voiding episodes), a step past its timeout, output or state over its bound, and a result that is not a `GradeResult` with a finite reward in `[0, 1]`. Return agent-attributable outcomes instead of raising: a sentinel from `extract`, `GradeResult(0.0, facts)` from `grade`. Raise `reliquary_sandbox.episode_task.EnvInfraError` for a failure that is yours, not the agent's: the episode ends `aborted` (logged at error level, counted per env as `env_infra_error`; in `grade` retried like an infrastructure fault). Raise it only from steps that run before any agent-controlled state is applied in the box (preparing the pristine box), or for your own invariants (decoding the state your `extract` wrote); anything the agent's state can cause must grade 0, or an agent could void its own failing episodes.
- Move file trees with `runtime.archive(paths, max_bytes)` and `runtime.restore_archive(data, roots)`, never with the image's own `tar`, or anything run as root, on state the agent controlled. Reading agent state with the box's own tools as the box user is allowed when grading bounds the impact: SWE's `extract` runs the image's `git` in the agent's box, as the box user, to take a diff; the agent controls that git (its config, hooks, objects), so the diff is untrusted, and grading is what makes it safe (it applies the diff in a pristine box and refuses paths outside the tracked tree). Restore vets and rebuilds the archive (no `..`, no member under a symlink, no FIFOs or devices, owners and setuid dropped, bounded sizes), clears each root first, and must be called once per root: a second restore into a root aborts the episode as an integration bug. Terminal-style tasks that restore into `/app` set `grading_workdir="/"`.
- Call `await runtime.stop_processes()` before reading a result that agent-influenced code could still rewrite (terminal tasks: after `test.sh`, before reading the reward). It kills every process in the box but PID 1, the entrypoint and the helper; processes that keep respawning grade 0 (`grading_box_failed`), a failure on our side aborts.
- A hook that calls a runtime member the sandbox does not offer (`prepare_uv_script`, `run_uv_script`, `open_process`, `run_background`) aborts the episode.

### Verifiers envs (the generic bridge)

A verifiers v1 Taskset needs no `sandbox_task`. The bridge derives the box from `TaskData` (`image` through the lock, `workdir` or `/`, `resources` cpu/memory/disk in GB, `timeout.setup` for preparation (at least 35 s: setup runs in a scoring process), `timeout.agent` for wall time, the larger of `timeout.finalize` and `timeout.scoring` for each grading step, at least 35 s (a scoring process needs 30 s left after the restore to start, plus a 5 s margin) and at most 810 s) and `Task.runtime_env()` (values must be literals of the task's data or config: a `${VAR}` template would resolve against the gateway's environment). `Task.setup` prepares the agent's box (a failure fails the open). At finish, `Task.finalize` runs in the agent's box (an exception grades 0 `extract_failed`, past the step deadline `extract_timeout`), (every process its commands leave is killed after each one), then every declared artifact root (`TaskData.artifacts`, relative to the workdir, no `exclude` patterns) and verifiers' `/logs/artifacts` is archived with the sandbox's `archive`; a missing required root grades 0 (`missing_artifacts`). Grading starts a pristine box from the same digest (the raw image: `Task.setup` does not run there), runs the task's optional `Task.grading_setup(runtime)` there (the env norm's place for preparing the box from the image: regenerating a task's data, stripping a repository's history; a failure is ours, `aborted`; it is killed 35 s before the step deadline, so a task with `grading_setup` needs a grading step of at least 70 s, else it is refused), removes absent roots, restores the others, runs `Task.score(trace, runtime)` with a trace that has no messages (rewards grade the final state) and a runtime that stops every process after each command, then `stop_processes`. The reward is the trace's weighted sum and must be finite in `[0, 1]` (else graded 0, `reward_out_of_range`). An exception from the env's scoring code once the agent's state is restored (any exception `Task.score` raises, a reward that did not run) is graded 0 `grading_failed` (never aborted: the agent's bytes can cause it; logged and counted per env): fix it, the conformance suite fails an env that does it on malformed state. Tasks with LLM judges, GPUs or no `@vf.reward` are refused. Env norm: every hook (`setup`, `grading_setup`, `finalize`, rewards) runs in a scoring process; a hook acts only through the box (what it changes on its trace or its task is dropped with its process); finalize still only runs box commands and writes files (parsing belongs in the reward, which sees only the restored artifacts). The env code that does not depend on the agent still runs in the controller when tasks load, trusted like the env's import: the Taskset's construction, `len`/`task_at`/iteration/`with_system_prompt`, `Task.hash`, `runtime_env`, `hooks`, the attribute reads of the declaration, the conformance cases and `pickle.dumps(task)`. Run the conformance suite ("Episode hosts: testing") before declaring an env.

### Tools

`reliquary-tools/1` is recorded in record 0. `bash` runs each call as a fresh `bash -c` in the task's workdir and returns stdout followed by stderr; `edit` replaces a string that occurs exactly once. They match the verifiers bash harness except:

- output beyond `EPISODE_OUTPUT_CAP_BYTES` is cut and flagged `truncated`;
- a blank or whitespace-only `command` is rejected with exit code 2, not executed;
- `edit` reads and writes raw bytes decoded as UTF-8, so CRLF is kept;
- a string that cannot be encoded as UTF-8 is a bad-arguments error;
- invalid UTF-8 in command output becomes U+FFFD;
- lines of runtime error text (`OCI runtime`, `urpc method`, `containerManager.`) are replaced with `[runtime output removed]` before signing.

A miner-side runtime must match these rules or its transcript will not reproduce the signed observations.

### Verifying transcripts

`reliquary_sandbox.attest.verify_transcript(transcript, directory, token_verifier, expected)` is a pure function of its inputs: no clock, network or state, depending only on the standard library and `cryptography`. `ok` means the transcript is internally consistent, signed by the listed machine key and bound to the token the validator issued. It does not mean the episode is payable. The caller must:

- pass every `Expected` field (hotkey, engagement, env, split, index, checkpoint), and a snapshot of seen session ids;
- check that the calls and observations match the proven tokens, and that the final state hash matches the submitted diff, which the verifier does not do;
- record the session id atomically with acceptance, so a session pays once;
- refuse submissions that arrive after `expires_at + GRADING_GRACE_S` (1800 s) by its own clock, since the verifier uses none;
- on a key compromise, set the key's `valid_until` to the earliest suspected time, because the machine key is looked up at record 0's `at`.

A task whose grading cannot fit that window (two grading timeouts plus margins must stay within 1800 s) is refused at open with `422`.

## Episode hosts: env packages and task images

How `reliquary-swe` and `reliquary-terminal` are served: as `verifiers:` entries through the bridge, with no per-env adapter. The packages' own rules (what they serve and refuse) are in `docs/env-norm.md` of reliquary-environments.

- **Install** each env as a wheel, never editable: the bridge's version gate needs the installed distribution to list the module. In the package's directory of reliquary-environments at the release commit run `uv build --wheel`, copy the wheel (and the dependencies from the package's lock, as for any env) to the host, and `uv pip install --reinstall <wheel>` into the controller's venv. The venv and the package files must be readable by the scoring slot uids (`chmod -R o+rX`, see "The scoring users" below).
- **Configuration:**

  ```sh
  EPISODE_ENVS='{"reliquary-swe": "verifiers:reliquary-swe==0.3.0", "reliquary-terminal": "verifiers:reliquary-terminal==0.3.0"}'
  EPISODE_ENV_OPTIONS='{"reliquary-swe": {"tools": ["bash", "edit"], "splits": {"train": {"split": "train"}, "r2e": {"split": "r2e"}, "polyglot": {"split": "polyglot"}}, "defaults": {"per_call_timeout_s": 600}}, "reliquary-terminal": {"tools": ["bash"], "splits": {"tmax_rl": {"split": "tmax_rl"}}}}'
  ```

  An env served to a validator's RL run uses exactly the options of the order's catalogue entry (TMax's RL tasks are the `tmax_rl` split, as above): the gateway publishes the sha256 of its options and the validator places sessions only on machines whose options digest is the entry's. Write the gateway's `EPISODE_ENV_OPTIONS` value for that env from the same options file the validator's `describe` read, never by hand; a gateway serving other options (for example `{"tmax": {"split": "tmax"}}`) gets no session of that env.

- **Caches readable by the slot uids.** `HOME` (and `HF_HOME`) of the controller point at a directory such as `/var/cache/reliquary` (`o+rX`), never under `/root`: the terminal env's task directories and the SWE datasets live there. Warm them before start: `python -c "from reliquary_terminal import tmax; tmax.download_source()"`; the R2E and polyglot datasets download on first use (about 4.4 GB for R2E).
- **Memory of the dataset loads.** Loading R2E-Gym-Subset needs more than 3 GB once: its cached parquet files are one row group each (up to about 747 MiB uncompressed, and pyarrow needs about 2.5 GB to read the largest), so a loader under a 3 GB cap is killed. Either give the first load of an episode host that much memory (the SWE-smith `train` listing peaks near 3 GB too), or rewrite each cached parquet file into small row groups (for example 8 rows; same rows, order and schema) once, after which loading fits in the cap. The `train` split is then served one row at a time (about 225 MB resident).
- **Images** come from the packages' locks (`scripts/sandbox_images.py from-lock` in reliquary-environments), pulled by digest before start; the TMax base image is about 10 GB.
- **Gate.** An env is declared only with a conformance report that passed as a gate (`reliquary-sandbox-conformance ... --out`), run with the env's served options and split. The report records the digest of the image lock the env was served with (`image_lock_sha256`, `ImageLock.sha256()`); a validator accepts it only for an env pinning that same lock. SWE's gate needs a host with room for the 32-task sample's images (one per task for R2E and polyglot).
- **Run every Python step under a memory cap on shared hosts**, for example `systemd-run --scope -p MemoryMax=3G -p MemorySwapMax=0 <command>`. This applies to cache warming, image listing and any test run.
- **Pre-pull and approve.** The gateway never pulls. `python scripts/sandbox_images.py pull <manifest.json>... --docker-host <host>` fetches the images, `check` runs each under the host's runtime (`--runtime runsc`, `--require git` for SWE images), and records each image's last result (`--record`, default `sandbox-images.checked.json`); `approve` prints the value to set as `RELIQUARY_SANDBOX_EPISODE_IMAGES` only when every image's last recorded check passed, and otherwise lists the others and exits 1. `scripts/sandbox_images.py` is in the environments repository.
- **Never run a second dockerd with `"bridge": "none"`**: moby deletes the host's `docker0` when it starts. See "Episode hosts: testing".
- **Task limits** are enforced at open. A session token whose `memory_bytes`, `disk_bytes` or `pids` budget is below the task's `TaskLimits` is refused with `422`, signed and spent (the validator chose the budgets). `cpus` sizes the box, capped by `EPISODE_NANO_CPUS`. The `EPISODE_MAX_*` settings must be at least what the served tasks declare (SWE: 4 GiB, 10 GiB disk, 1024 pids, 3600 s wall, 600 s per call).
- **Grading timeouts** must satisfy `2 * grading_timeout_s + 180 <= 1800`: the packages declare 810 s (SWE) and 600 s (terminal).
- **The scoring users (verifiers bridge).** Every env hook of a bridged env (`Task.setup`, `Task.grading_setup`, `Task.finalize` and the reward code) runs in a scoring process started from a forkserver of the controller. Requirements, all checked by the code (a missing one ends every bridged grade `aborted`, never a scoring process with the controller's privileges or secrets):
  - **Root, with `CAP_SETUID`, `CAP_SETGID` and `CAP_CHOWN`** (`compose.episodes.yaml` adds them; `no-new-privileges` stays on). A controller that does not run as root refuses to score: it logs `verifiers scoring is refused` at start and every bridged grade is `aborted`. `RELIQUARY_SCORING_UNPRIVILEGED_OK=1` lifts this for tests only (the scoring process then keeps the controller's uid, and other episodes and the controller's files are not protected from it).
  - **A dedicated uid range.** Each scoring process runs as the uid:gid of its slot, `SCORING_UID_BASE + slot` (default `200000`, slots `0` to `SCORING_SLOTS - 1`, default 64: uids 200000-200063), with no supplementary group and `no_new_privs`, so code execution through an env's parser cannot open the Docker socket, the state directory or the machine key, nor signal or read another episode's scoring process. Nothing else on the host (or in the controller's user namespace) may use that range: not a system account, not a userns-remap `subuid` range (Docker's default is 100000-165535), not another controller sharing the pid namespace. `SCORING_SLOTS` must be at least `EPISODE_CAPACITY` (startup refuses a configuration with a `verifiers:` entry and fewer slots); a grade that finds no free slot while 30 s or more remain waits, then is `aborted`.
  - **What runs after each grade.** Once the scoring process is reaped, a sweeper process (from the forkserver, dropped to the slot's uid) sends `SIGKILL` to every process of that uid (`kill(-1)`: whatever the env's code left running, double-forked or in its own session) and empties the scoring TMPDIR, which the controller then removes (a slot uid cannot unlink an entry of the root-owned `/scoring`); only then is the slot handed to another episode. It also removes the uid's entries in `/dev/shm` (POSIX shared memory and semaphores; the directory is sticky, so only the uid itself can) and gives a directory the env's code locked (`chmod 000`, read-only) back `0700` before removing it. A sweep that fails or takes more than 30 s keeps its slot (logged `could not be swept`) and is retried at a later grade: each grade retries at most one pending sweep before its own, so a grade's overrun stays bounded (two sweeps, 60 s). State that could outlive the sweep elsewhere is closed by `compose.episodes.yaml`: the service's `/tmp` tmpfs is `mode=0700` (a slot uid cannot write there) and `sysctls: kernel.shm_rmid_forced: 1` destroys a SysV shared memory segment once no process is attached to it (one never attached, when its creator exits) (SysV message queues and semaphores are not covered; a scoring process can create them, bounded by the IPC namespace's limits). Under host memory pressure a scoring process is killed first: it sets its `oom_score_adj` to 1000 before it drops its privileges (best effort, a failure is logged). The scoring process also runs with `RLIMIT_NPROC` 64 on its slot's uid (processes and threads), `RLIMIT_FSIZE` `BridgeDefaults.scoring_file_bytes` (64 MiB: a larger file fails with `EFBIG` in the env's code), `RLIMIT_AS` (`scoring_memory_bytes`, 2 GiB over what it maps) and `RLIMIT_CPU` (the step's time left).
  - **No network.** After `no_new_privs`, the scoring process installs a seccomp filter: the filter is installed before any code of the env runs in the scoring process; import-time code of a preloaded env runs in the forkserver as root, with network, and is trusted like the env's task loading. `socket` of any family but `AF_UNIX` (`AF_INET`, `AF_INET6`, `AF_NETLINK`, `AF_PACKET`, `AF_VSOCK`, ...), `io_uring_setup`, `ptrace`, `process_vm_readv`, `process_vm_writev` and `pidfd_getfd` fail with `EPERM`, and a syscall of another ABI (i386/arm compat, x32) kills it; everything it starts inherits the filter. It talks to the controller over the pipe it already holds. The filter exists for x86_64 and aarch64 (neither has `socketcall` natively); on any other architecture, with a 32-bit interpreter, or when the kernel refuses the filter, the scoring process refuses to score (logged `refuses to score without its network filter`) and every bridged grade is `aborted`.
  - **Controller limits** (`compose.episodes.yaml`): `pids_limit: 1024` (the service's threads plus `EPISODE_CAPACITY` x 64 scoring tasks; raise it with the capacity) and `mem_limit: 12g` (the service, the 512 MiB scoring tmpfs, and `EPISODE_CAPACITY` x the 2 GiB scoring budget; resize both with `EPISODE_CAPACITY`).
  - **The wrapper** `/usr/local/libexec/reliquary-scoring-python` (`SCORING_WRAPPER`; the image installs `scripts/scoring-python` there, mode 0755, root-owned, outside `/tmp`, which is mounted `noexec`) starts the forkserver's Python with an empty environment (`env -i`, then only `PATH`, `LANG`, `PYTHONHASHSEED=0` (every scoring process hashes strings the same way on every validator and after every restart) and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`: math libraries run on one thread, within `RLIMIT_NPROC`), so no secret of the controller is ever in the forkserver's memory, hence in no scoring process. It runs `/app/.venv/bin/python` (`RELIQUARY_SCORING_PYTHON` overrides it, tests). A missing or non-executable wrapper fails the forkserver's start.
  - **Preloaded env modules.** The forkserver runs only when `EPISODE_ENVS` has a `verifiers:` entry (a configuration of `module:callable` envs alone starts none, and the scoring path never imports their modules). It imports the module of every verifiers entry when it starts (as root, with that empty environment), so scoring processes import nothing heavy. A scoring process must be ready within `BridgeDefaults.scoring_ready_s` (120 s; it imports the env's module itself when the forkserver did not) and is never started with less than 30 s left before the step deadline (graded 0 `grading_timeout`); past the ready limit the grade is `aborted`. An env module must not need a variable of the controller's environment at import time, nor at scoring time: **an env must read no credential at import time**. The forkserver skips a module whose import fails, so the controller checks them when it starts: it imports `PRELOAD` and the module of every verifiers entry the same way (through the wrapper, empty environment, the controller's `sys.path`) and refuses to start when one fails (`RuntimeError: the scoring forkserver cannot preload the env modules ...`, with the import's error).
  - **Scoring TMPDIRs** live under `SCORING_TMP_ROOT` (`compose.episodes.yaml`: a 512 MiB `/scoring` tmpfs with at most 65536 inodes, `noexec`, apart from the service's `/tmp`), one private `0700` directory per grade owned by the slot's uid. Each file is capped by `RLIMIT_FSIZE`; the total is shared by the episodes grading at once. Where the kernel supports tmpfs user quotas (Linux 6.6+), a host-mounted tmpfs with `usrquota,usrquota_block_hardlimit=64m` bind-mounted at `/scoring` gives each slot uid its own quota.
  - **Readable files.** The scoring process imports the env's modules (when not preloaded) and reads the task's files as its slot's uid: the controller's venv and site-packages, the env packages, and every task file the reward code opens (the terminal package's materialized tests: point `HOME` / the cache at a directory such as `/var/cache/reliquary`, not under `/root`, which is `0700`) must be readable by others (`o+r` on files, `o+x` on every directory up to `/`). Check with `setpriv --reuid=200000 --regid=200000 --clear-groups env -i HOME=/nonexistent PATH=/usr/bin:/bin /app/.venv/bin/python -c "import reliquary_terminal"` (the scoring environment: an env such as verifiers' Harbor module calls `Path.home()` at import, which fails for a slot uid without `HOME`, having no passwd entry) and by reading a task file the same way. Its environment is only `PATH`, `LANG`, `HOME=/nonexistent`, `TMPDIR` and `PYTHONHASHSEED=0`, plus the three thread variables `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` kept from the scoring environment: an env must resolve paths when its tasks load in the controller, never from another variable at scoring time.
  - **No code from the agent's bytes.** Once a scoring process has unpickled its task and set its limits, and before it says it is ready (before any env code that sees the agent's state runs), it installs a Python audit hook (`sys.addaudithook`, which cannot be removed). The hook refuses `exec`, `compile` (except `ast.parse`, which only builds a syntax tree, and `typing.ForwardRef` resolving a string annotation of the env's source), `code.__new__`, `marshal.loads` / `marshal.load`, `pickle.find_class` (any unpickling that names a global), every `ctypes.*` event, `subprocess.Popen`, `os.system`, `os.exec*`, `os.posix_spawn*`, `os.spawn*`, `os.fork` and `os.forkpty`. Imports keep working: while importlib loads a module of the installation, its `exec`, `compile`, `marshal.loads`, `code.__new__` and `ctypes.*` events are allowed, including `dataclass` and `namedtuple` at module level. Processes, unpickling and string code are refused even then. A refusal raises `PermissionError` where it happens and is remembered for the rest of the process: the hook's result is `raised` (graded 0 `grading_failed`) even when the env's code catches the exception and returns a reward. So a parser exploited through `eval`, `pickle` or `ctypes` cannot run the agent's code to forge the result. The SWE (0.3.0) and terminal (0.3.0) grading paths need no exemption beyond imports, `ast.parse` (the terminal env reads its tests' syntax tree) and `typing.ForwardRef` (pydantic resolves postponed annotations the first time the terminal env's `grading_setup` builds its verifier task). An env whose reward relies on sympy's or math_verify's parsers (`parse_expr` evaluates strings), on unpickling or on a subprocess of the scoring process grades 0 on everything: run its commands in the box instead.
  - **Exit codes.** After it said it was ready, a scoring process that ends without a result is the scoring's (graded 0 `grading_timeout`), whatever its exit code, 70 and 71 (our own failure codes) included: code the agent's bytes steer can exit with any code. A protocol failure of ours is detected by the controller itself: it checks every reply against what the scoring process reads and ends the grade `aborted` before sending one it could not read. Box and runtime faults keep their meaning (`grading_box_failed`, infra errors).
  - **Residual.** The audit hook covers the usual ways a parser turns data into code, not a deserializer that calls arbitrary callables with arguments it builds. PyYAML's unsafe loaders are the main example: `python/object/apply` can reach `os.write` on the scoring process's pipe, which no audit event covers. Code that gets that far can still forge its result, which is the scoring process's word. The conformance suite therefore fails an env whose package imports `math_verify`, `sympy`, `latex2sympy2`, `pickle` (or `dill`, `cloudpickle`, `shelve`, `marshal`) or uses an unsafe PyYAML loader (`yaml.load`, `full_load`, `unsafe_load`, `Loader`, ...; `safe_load` is fine). The report lists them in `unsafe_imports`, and each is a suite problem marked `requires review`. The check is static: it reads every import statement in the package's own source files, including lazy ones inside functions, but not imports built at run time or made by a dependency. What is enforced whatever runs: isolation (no secret, no root, no network, no other episode), and no exit code voids an episode once the scoring process is ready.
- **Terminal (tmax) grading** runs the task's test as an unprivileged uid inside the grading box and writes the verdict as root after killing every process of that uid (reliquary-environments `docs/env-norm.md`). What remains is code the test starts tampering with the test process of its own uid where the kernel allows it.

## Episode hosts: runtime start failures

Under runsc, a loaded host sometimes fails to start a command in a box that is otherwise healthy. Docker reports it as an exec that exits with a code of 126 (Docker 23 and older) or 128 (measured live on Docker 29, where the exec record shows `"Pid":0`: no process was ever started), with exactly this text and nothing else on the stream:

```
OCI runtime exec failed: <path>/runsc did not terminate successfully: exit status N
```

Only that exact message counts (anchored at both ends, nothing on stderr, no helper frame marker); text an agent or helper echoes never does. `<path>` is `.../runsc`, or the wrapper dockerd generates when the daemon configures `runtimeArgs` for the runtime: `<data-root>/runtimes/runsc.<hash>`. The detector accepts that wrapper. Keep the required `--overlay2=none` argument and requalify any additional runtime arguments on the test host (`--ignore-cgroups` is refused outright).

What the service does with it:

- It is a `RuntimeExecFailed` (a `BoxFault`): the host's, not the agent's, when the box still runs and is not at its pids limit.
- Read-only helpers are run again in place (up to 2 more times, 1 s apart).
- A grading attempt that hits it is re-graded in a fresh box (`EPISODE_GRADING_ATTEMPTS`, default 2, time permitting). When every attempt hits it, the episode keeps the `grading_box_failed` zero it would always have had, marked `runtime_exec_failed: true` in the `_grader` fact of the final record. It is never voided: the 0 is kept.
- In the agent's box at extract it ends `box_failed` (unpaid, not voided); the conformance suite runs such a case again on its side and fails the suite (`ConformanceInfraError`, exit 2) if the host keeps failing it.

Monitor the count of final records with `runtime_exec_failed` per host, and the `Error running exec` lines of dockerd's log. A rise means the host is overcommitted (the failures clustered on a 2-vCPU host asked for 16 CPUs): lower the load, do not raise the retries. The conformance report records, per run and in total (`regraded`, `helper_retries`, also printed on stderr), the fresh-box re-gradings and in-place helper retries even when every case passes, so a PASSED report shows what the runtime cost.

## Episode hosts: the env options digest

The capacity report publishes, per bridged env, `env_options_sha256`, and a validator that builds the same env refuses to place a session on a machine whose digest differs. Reading a refusal:

- The digest covers the canonical JSON of the whole options in effect, defaults filled in: `tools`, `splits`, `defaults` (`BridgeDefaults`) and `publish_state`, including the gateway's own scoring settings that live in `defaults` (`scoring_memory_bytes`, `scoring_ready_s`, `scoring_file_bytes`). Tuning one of those on one gateway therefore refuses placement until the validator's copy of the options matches. Strict equality is intended.
- The comparison is on values as written: `600` and `600.0` give different digests for the same field. Write integers as integers on both sides.
- On a mismatch refusal, compute the digest from the gateway's `EPISODE_ENV_OPTIONS` and from the validator's copy (`options_sha256(...)`) and diff the two canonical JSON documents; the differing key is the cause (usually a default tuned on one side, a float, or a tool list). The identity (`env_package`, with `.vfb<bridge version>`) is a separate check: a bridge version bump changes it, and both sides must move together.

`env_image_locks` (bridged envs only): per env, the sha256 of its image lock as served (`ImageLock.sha256`: the canonical JSON `{"version":1,"images":{...}}`, sorted keys, no spaces), how many distinct images it pins, and how many of them this machine holds. "Held" means listed in the report's `images`: pulled and approved (`RELIQUARY_SANDBOX_EPISODE_IMAGES`); an image pulled but not approved does not count. A validator activates an env added to a running run only when a machine reports its lock with enough of its images present (all of them for a small lock; a threshold for an env with one image per task, whose servable tasks validators compute from the `images` list). The block is absent on a machine serving no bridged env.

## Episode hosts: gateway runbook for RL capacity

- **Envs with one image per task** (R2E, polyglot): a validator places a session, and accepts a precommit, only for a task whose image a serving machine holds (its `images` list). Pre-pull, on each machine, the images of the tasks you mean to serve (`docker pull` of the lock's digests through the isolated daemon, then approve them as in "Pre-pull and approve" above), and keep the activation threshold on at least one machine before an env is added to a running run: both thresholds apply on one machine (at least 32 locked images AND at least 256 servable tasks of the RL split, on the same machine). The service lists the local images once per 45 s (one `images.list()` call, not one lookup per image), so an image pulled outside the service shows in the machine's `images` within 45 s; an image removed outside the service may still be listed for up to 45 s (an open of its task then fails to create the box and the next lookup lists again).
- **A new version of an env** is a new distribution under a new sandbox env name (never an in-place upgrade while the old one is active): install it beside the old one, add the new name to `episode_envs` / `episode_env_options`, and restart the gateways one at a time (rolling), checking each machine's report lists both names (in `env_packages`, `env_options_sha256` and `env_image_locks`) before the next. Only then may the validator's operator request the addition.
- **A grading waiting for the pool keeps its agent box and its session slot** (and the agent box's reservation) until it enters the pool or is aborted. A full grading pool therefore also holds open slots and reserved resources: watch `grading_pool.waiting` in the signed report alongside `grading_pool.timeouts`; a `waiting` that stays above zero means the pool is undersized for the declared capacity.

## Episode hosts: runsc version check

With episodes enabled, preflight runs `runsc --version` from the runtime path that `docker info` reports
and compares it to `EPISODE_RUNSC_VERSION` (default `release-20260928.0`); `--ignore-cgroups` in the runtime
arguments is refused. The path must therefore be visible to the controller: a controller running in a
container needs the host's `runsc` mounted at the same path, or the check fails closed. The per-box
`/etc` files are written under the state directory, which must also be a host path the Docker daemon can bind-mount.

## Episode hosts: readiness and capacity

The episode host checks (the preflight above, without the quota probe) run every
`EPISODE_HOST_CHECK_SECONDS` (default 30) and gate the whole node's readiness: when they fail, `/readyz`
answers 503 for the tenant API too. `/readyz` serves the last verdict and never runs the checks itself.
`GET /capacity` serves the signed capacity report (cached for 1 s); it is also pushed to `HEARTBEAT_URL`
when set. Validators take a machine's address from their directory, never from a report.

`episode_cpu_millis_total` and `episode_memory_bytes_total` (both or neither): the CPU (thousandths) and memory all live episodes' agent boxes may reserve. Each episode reserves its agent box: its task's CPUs and its memory budget. The reservable CPU is the total × `episode_cpu_overcommit` (1.0 to 4.0, measured at qualification: an agent box idles while the miner runs its model); memory is never overcommitted. An open that does not fit is refused 503 `no free resources`, one that never fits 422; on such a machine a task asking more CPUs per box than `episode_nano_cpus` is refused 422 instead of clamped. The grading boxes run in the grading pool (`episode_grading_*`, required with these totals). Set `episode_cpu_millis_total + episode_grading_cpu_millis_total` at or below the host's CPUs × 1000 and the two memory totals below the host's RAM minus the service's own needs. These totals require `episode_present_window_s` (10 to 120, the validator's present window: it stops holding an unpresented token's place after 120 s); the service refuses to start without it. The signed capacity report then carries a `resources` block (reservable totals, used, free, the per-box CPUs, the overcommit, `present_window_s`, the live session ids); validators place RL sessions only on machines that report it, and a machine that serves RL sessions must not also be placed on by a corpus validator.

`episode_grading_cpu_millis_total` and `episode_grading_memory_bytes_total` (both or neither; required with the reservation totals): the grading pool. Each grading attempt's box (the task's CPUs and the token's memory) runs only inside it, entered after extract; waiters are served fairly by hotkey (the one holding the least of the pool first). A grading that cannot start while its grade step still fits is aborted by the machine: `the grading pool stayed full` when other miners' gradings held the pool (validators charge nobody and count it toward no cap), `the grading pool stayed full with this miner's own gradings` when the session's own hotkey held more than half of it (an ordinary abort). Size it at qualification so that `grading_pool.timeouts` stays at zero at the declared capacity; the signed report's `grading_pool` block shows its use, its waits and its timeouts.

A new episode is refused before preparation when its grading box could never fit the configured grading pool. Closing an episode does not release resources still occupied by a container whose removal failed: agent reservations and grading pool shares remain held until cleanup succeeds, including across controller restarts. On slot-only hosts, any leftover episode box keeps its slot occupied. The periodic cleanup retries removal and releases each hold once.

## Episode hosts: the helper busybox

Every plumbing command inside a task box (the bash runner, the kill, file reads and writes, the type check,
the user lookup) runs from a static busybox that the controller bind-mounts read-only at
`/.reliquary/bin/busybox`; the image's own `sh`, `head`, `cat` and `stat` are never used, so an agent cannot
break the plumbing by replacing them.

* Obtain it from the Debian/Ubuntu `busybox-static` package (`apt-get install busybox-static`, then copy
  `/bin/busybox` to a root-owned, mode 0755 path such as `/opt/reliquary/busybox`). It must provide the
  applets `sh`, `head`, `cat`, `mkfifo`, `mkdir`, `chmod`, `rm`, `kill`, `stat`, `id`, `sleep` (fractional) and `setsid`.
* Set `EPISODE_HELPER_BUSYBOX` to that path and `EPISODE_HELPER_BUSYBOX_SHA256` to `sha256sum` of the file.
  Both are required when episodes are enabled; preflight and every box creation refuse a missing,
  non-executable or modified file.
* The controller copies the verified binary into `<state_dir>/episode-helper/bin/busybox` and bind-mounts that
  whole directory read-only at `/.reliquary`, so the binary cannot be renamed or replaced from inside the box.
* To attribute a stopped box to a daemon restart (an infrastructure fault, so the episode is aborted rather than
  charged to the miner), the controller reads the daemon start time from `EPISODE_DOCKER_PIDFILE`
  (default `/var/run/docker.pid`) and `/proc`. If either is not visible to the controller, preflight refuses to
  enable episodes: run the controller on the host, or mount both read-only.
* Size memory so the host never overcommits: the sum of the boxes' memory limits plus the controller must stay
  within host RAM. A host-wide OOM is a capacity failure on our side, but when `memory.events` is unreadable it is
  charged to the agent.
* Size CPUs the same way: each episode may run its agent box and a grading box at once, so
  `EPISODE_CAPACITY x 2 x EPISODE_NANO_CPUS / 1e9` should not exceed the host's CPUs. Preflight logs a warning and
  reports `cpu_overcommitted` (with `host_cpus`) in its facts when it does; calls and grading then run slower than
  their budgets assume.
* A box that still reads `running` after a failure is re-checked after `EPISODE_FAULT_SETTLE_S` (default 1 s)
  before the failure is charged to infrastructure, so a box dying under us is not misattributed.
* An out-of-memory kill is charged to the agent only if the box's own cgroup limit was hit (`memory.events`
  `oom`). That file is read under `EPISODE_CGROUP_ROOT` whatever `EPISODE_CPU_SOURCE` is, so the host's cgroup v2
  tree must be visible to the controller; when it is not readable, the kill is charged to the agent.

## Episode hosts: behaviour measured under runsc

Measured live (Docker 29.1.3, runsc `release-20260928.0`, busybox-static 1.37, `ubuntu:22.04`):

* **CPU.** Both `EPISODE_CPU_SOURCE=docker` (one-shot stats) and `cgroupfs`
  (`system.slice/docker-<id>.scope/cpu.stat`) report runsc boxes' CPU, Sentry included, and never decrease.
  Attributing an OOM kill by `memory.events` needs the host cgroup tree under `EPISODE_CGROUP_ROOT`, with either source.
* **Memory.** A box over its memory limit is OOM-killed as a whole (`memory.events` `oom 1`, `oom_kill 1`,
  `OOMKilled: true`). A fork bomb at the pids limit and an agent's `kill -9 -1` also take the box down. Docker may
  still read `running` for a moment after the sandbox stopped, and an exec then fails with runtime text
  (`OCI runtime exec failed ... in state stopped`, `urpc method "containerManager.WaitPID" failed`) and exit 128.
* **The runner's trailer.** The bash runner's last write is `RQ-END <call id> <status>` on its own stdout, which the
  command never holds. The controller strips it and requires it, as the final bytes, with the exec's exit code. When
  it is missing or disagrees, and also (since a trailer can be forged) whenever the exit is >= 126 or the output
  carries runtime-looking text, the box is probed with a helper `true`; only when that fails is the settle period
  paid before a second reading and probe. A box that no longer runs a
  helper fails the episode AT that call (`box_failed`; `aborted` only when `classify_failure` finds a host cause) and
  the call is not recorded. On a live box the outcome is always the agent's: the call is recorded with its exit code.
  Runtime-looking lines (`OCI runtime`, `urpc method`, `containerManager.`) are never signed: they are replaced by
  `[runtime output removed]` in every call output, so an agent cannot void its episode by printing them. A live box
  answers the probe in one exec (a raw exec is about 60 ms), so an ordinary exit 127 costs no settle period.
* **Exec output and the transport.** docker-py's `exec_start(socket=True)` returns the socket under the reader that
  parsed the `101 UPGRADED` headers: a first output frame that arrives with the headers is silently lost (about 2 %
  of fast execs on the test host; a bug in our client, not in gVisor). The controller therefore performs the exec-start
  upgrade itself on the daemon's Unix socket (`episodes/exec_transport.py`) and keeps those bytes; a `DOCKER_HOST`
  that is not a Unix socket is refused at preflight. Live: 0 lost in 300 fast execs. As defence in depth, helpers
  whose output is parsed (`read_file`, `tar_create`, `resolve_existing`) print it between markers carrying a
  per-exec random nonce and are re-run when a marker is missing, and env hook programs (`LocalRuntime.run`) run
  framed the same way but are never re-run. An output still missing (or a failed re-run) is never an empty result:
  it is classified like any failure. With the transport fixed, a box that still runs lost it through the agent
  (killed or starved the helper), so it is a `BoxFault`; on a stopped box the host rules above decide.
* **busybox-static prefers its applets.** In its `sh` and `env`, a bare `ls` or `grep` runs the busybox applet, not
  the image's binary. The program runner therefore looks names up on `PATH` itself and runs the image's file by path.
* **Timeouts.** A timed-out call returns after about its timeout, plus the kill, plus up to 5 s of drain (about
  1.8 s measured); the output the command printed before the timeout is kept (the runner's `head` buffers it until
  it drains).
  A call that leaves a background job or a `setsid` escapee returns in about 1.8 s.
* **Disk quota.** With runsc `--overlay2=none` on Docker overlay2 over xfs `pquota`, a 256 MiB box stops at its quota with `ENOSPC` inside the box (the
  Sentry keeps running), deleting frees the space, and the read-only `/etc/hosts`, `/etc/hostname`,
  `/etc/resolv.conf` and `/.reliquary` binds refuse writes (`EROFS`, `EBUSY` on rename, `ETXTBSY` on overwrite).
* **Cost.** Median on a 2 vCPU host shared with another workload: open (create + prepare) 0.6 s, bash call 0.26 s,
  raw helper exec 60 ms, finish (extract + pristine-box grading) 1.0-1.2 s, idle box memory about 20 MB.

## Episode hosts: testing

The env conformance suite gates a verifiers env. It runs the env through the bridge on a local gateway and checks the env's reference solutions grade 1, its declared attacks and the built-in ones (untouched box, empty and malformed artifacts, deleted artifacts, artifacts replaced by a link to `/`, links into `/logs/verifier` and `/tests`, planted reward files, shimmed tools, a background reward writer) grade 0, two concurrent runs of each case agree, every transcript verifies and nothing ends `aborted` or `box_failed`. A zero the env's own code caused (`extract_failed`, `grading_failed`, ...) fails the suite, except `extract_failed` on the built-in shimmed-tools case: an extract that runs the agent's own (sabotaged) tools, as verifiers' Harbor `collect` does, gives the agent its own 0. An env declares its cases as plain data in its top-level module, `conformance_cases(split) -> [{"name", "index", "expect": 1.0 | 0.0, "calls": [[tool, arguments], ...]}]`; `--cases module:callable` supplies them for a third-party env. The untouched/empty-artifact attacks run on a random sample of the split; the report records its `sample_salt` and `sample_seed`. `--sample-size N` (0–32) reduces the sample for a host without room for that many task images (one image per task in SWE's R2E and polyglot splits); such a run is not a gate (`gate_reasons: ["sample"]`). On the test box, against the isolated daemon with the env's images pulled:

```bash
eval "$(scripts/test-xfs-docker.sh env)"
uv run reliquary-sandbox-conformance --entry 'verifiers:<taskset-id>==<version>' --split rl \
  --docker-host "$DOCKER_HOST" --docker-pidfile "$EPISODE_LIVE_DOCKER_PIDFILE" \
  --busybox /usr/bin/busybox --out conformance.json
```

The suite also reads the env's package statically and fails it when it imports a parser that turns the agent's bytes into code (`unsafe_imports` in the report, each a `requires review` problem: see "The scoring users", Residual). Exit 0 means passed, 1 failed, 2 not runnable; keep `conformance.json` as the attestation the operator signs off. A real run refuses an unpackaged taskset. `--fake` runs the same cases over FakeBoxes (no Docker, unpackaged tasksets allowed) as a smoke test of the cases, not a gate; its report says `"fake": true`. `tests/test_bridge_live.py` runs the suite on a toy env and on verifiers' unmodified `HarborTask` (opt-in like the other live tests; run it as root so the scoring processes take their slot uids, from a venv and Python the slot uids can read).

`tests/test_episode_live.py` runs the episode boxes on a real Docker host with gVisor. It is opt-in
(`RUN_EPISODE_LIVE=1`, marker `live`) and must never run on a production machine. Run it sequentially, as root on
the test host, once against the main daemon (whose storage is refused by preflight; the tests set
`episode_allow_unquoted_disk`) and once against the isolated xfs daemon, which also runs the quota test:

CI also runs `test_episode_resources_live.py` and `test_bridge_live.py` against the isolated XFS daemon as root, with project quotas and cgroup accounting enabled. This job fails if any selected live test is skipped. It separately runs `test_verifiers_bridge_live.py`, using the real miner harness with a local scripted model and signed gateway. These test hosts use temporary state and generated keys; they do not connect to production gateways.

```bash
RUN_EPISODE_LIVE=1 uv run pytest -q -s -m live tests/test_episode_live.py
sudo scripts/test-xfs-docker.sh up            # loop-mounted xfs with pquota under /opt, own socket and pidfile
docker -H unix:///run/reliquary-xfs-docker.sock pull ubuntu:22.04
eval "$(scripts/test-xfs-docker.sh env)" && RUN_EPISODE_LIVE=1 uv run pytest -q -s -m live tests/test_episode_live.py
sudo scripts/test-xfs-docker.sh purge         # stop it, unmount, delete what `up` created
```

The second daemon shares the host's containerd under its own namespace and uses its own private bridge `rqxfs0`
(10.231.0.1/24) with iptables, ip-forward and masquerading off. Never configure it with `"bridge": "none"`: moby
then deletes the host's `docker0` on start, which cuts the main daemon's container networking on the
test host. `up` snapshots the host's links and tears the daemon down if `docker0`
or any other pre-existing link changed; `down` signals only a pid whose command line carries this daemon's config
file, and deletes `rqxfs0` only if `up` created it (a marker file in `/run`). After every live step, check that `docker0` still exists. Set `EPISODE_LIVE_CPU_SOURCE=cgroupfs` to test the cgroupfs
CPU source and `EPISODE_LIVE_METRICS=<file>` to collect the measured values as JSON lines.

`reliquary-sandbox-reference-sweep` grades an env's own reference solutions (`reference_calls(split, index)` in its top-level module) over a range of tasks, one run each, and writes one JSON line per task (with the run's provenance): a measurement for qualification (how many honest solutions grade 1 on gVisor), not a gate. Same Docker flags as the conformance CLI; `--fake` for a smoke test.
