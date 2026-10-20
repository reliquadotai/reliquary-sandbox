# Reliquary Sandbox

On-demand code execution for miner workloads, agents, and evaluations. A self-hosted API creates isolated Linux containers, executes commands, exchanges files, and restores bounded workspace checkpoints. The Python SDK includes synchronous and asynchronous clients, typed models, bulk operations, retry handling, and the Connect RPC command format.

The runtime uses Docker with gVisor `runsc`. It requires a dedicated Linux host with Docker and Linux 5.6 or newer. The controller requires tenant credentials and an approved image list, and refuses to create sandboxes when the required runtime is unavailable.

## Start locally on Linux

1. Install Docker Engine and its Compose plugin. Install the pinned runtime:

   ```bash
   sudo ./scripts/bootstrap-runtime.sh --configure-docker
   ```

   This registers `runsc` and reloads Docker. Read the [gVisor installation guide](https://gvisor.dev/docs/user_guide/install/) for host requirements.

2. Build the guest and controller images:

   ```bash
   docker build --target guest -t reliquary-sandbox-guest:0.1.0 .
   docker build --target controller -t reliquary-sandbox-controller:0.1.0 .
   ```

3. Generate two tenant credentials in the ignored `.env` file:

   ```bash
   umask 077
   python3 - <<'PY'
   import json
   import secrets
   from pathlib import Path

   credentials = {name: secrets.token_urlsafe(32) for name in ("evals", "miners")}
   Path(".env").write_text("RELIQUARY_SANDBOX_TOKENS=" + json.dumps(credentials) + "\n")
   PY
   ```

4. Start the service and check readiness:

   ```bash
   docker compose up -d --wait
   curl --fail http://127.0.0.1:8000/readyz
   ```

5. Install the SDK and run an evaluation:

   ```bash
   uv sync --all-extras --frozen
   export RELIQUARY_SANDBOX_API_BASE_URL=http://127.0.0.1:8000
   export RELIQUARY_SANDBOX_API_KEY="$(python3 -c 'import json; from pathlib import Path; print(json.loads(Path(".env").read_text().splitlines()[0].partition("=")[2])["evals"])')"
   uv run python examples/eval.py
   ```

The Compose service binds to loopback. See [deployment](docs/deployment.md) before making it reachable by miners or other machines.

Signed episode hosts for agentic environments (signed tool outputs, pristine-box grading, offline transcript verification) are described in [deployment: episode hosts](docs/deployment.md#episode-hosts) and [compatibility: signed episodes](docs/compatibility.md#signed-episodes).

## Python usage

```python
from reliquary_sandbox import APIClient, CreateSandboxRequest, SandboxClient

api = APIClient()  # reads RELIQUARY_SANDBOX_API_KEY and RELIQUARY_SANDBOX_API_BASE_URL
sandboxes = SandboxClient(api)
sandbox = sandboxes.create(CreateSandboxRequest(
    name="code-eval",
    docker_image="reliquary-sandbox-guest:0.1.0",
    cpu_cores=1,
    memory_gb=1,
    timeout_minutes=10,
))
try:
    sandboxes.wait_for_creation(sandbox.id)
    result = sandboxes.execute_command(
        sandbox.id, "python -c 'print(sum(range(100)))'", timeout=30
    )
    assert result.exit_code == 0
    print(result.stdout)
finally:
    sandboxes.delete(sandbox.id)
    api.client.close()
```

Use a different credential per tenant. The same API supports `AsyncSandboxClient`, and SDK clients can use another compatible deployment by selecting its API origin and runtime mode. See [API compatibility](docs/compatibility.md) for the supported server surface.

## Command line

The CLI uses the same API URL and credential environment variables:

```bash
uv run reliquary-sandbox create --name evaluation --wait --json
uv run reliquary-sandbox run SANDBOX_ID 'python --version'
uv run reliquary-sandbox checkpoint SANDBOX_ID --wait --json
uv run reliquary-sandbox checkpoints SANDBOX_ID --json
uv run reliquary-sandbox restore CHECKPOINT_ID --name restored-evaluation --wait --json
uv run reliquary-sandbox delete SANDBOX_ID
uv run reliquary-sandbox delete-checkpoint CHECKPOINT_ID
```

Replace the IDs with the values returned by create and checkpoint. Restore keeps the checkpoint's image and workspace size; `--cpu`, `--memory`, and `--timeout` configure the new container. A retained checkpoint remains available after its source sandbox is deleted. The `run` command returns the guest command's exit status, capped at 125.

## Runtime capabilities

| Capability | Self-hosted service |
| --- | --- |
| Create, get, list, batch status, delete | Supported |
| Synchronous and asynchronous SDK | Supported |
| Commands through Connect RPC `Start` | Supported |
| File upload, download, and text read | Supported within `/workspace` |
| Tenant credentials, scoped gateway tokens, quotas | Supported |
| Approved images, CPU, memory, process and lifetime limits | Supported |
| Workspace checkpoint and restore | Bounded filesystem snapshot; no process state |
| Public network access | Disabled for guests |
| Managed secret injection | Unavailable; workload environment variables supported |
| VM provisioning, GPU devices, SSH | Unavailable |
| Image build service, registry publishing, billing | Unavailable |
| Background process sessions and reconnectable streams | Client interfaces included; unavailable on this server |

The SDK's broader client interfaces do not imply that every endpoint is provided by this deployment. Unsupported requests are rejected rather than silently downgraded.

## Project history

[0xgrizz](https://github.com/0xgrizz) delivered the initial Reliquary Sandbox implementation and `v0.1.0` release on 4 October 2026, including the SDK adaptation, CLI, self-hosted gVisor service, deployment tooling, and tests. The [original nine commits](https://github.com/reliquadotai/reliquary-sandbox/commits/66804e438538a9ab1451e3dfb9c7dcd9e6909757) retain their original authorship, dates, and IDs. Later work, including signed agentic episodes and the verifiers bridge, retains its contributor attribution in Git. Third-party licenses and copyright notices remain in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Development

```bash
uv sync --all-extras --frozen
uv run ruff check .
uv run pytest -q -m "not live"
```

On a Linux host with the guest image and `runsc` installed:

```bash
RUN_SANDBOX_LIVE=1 uv run pytest -q -m live tests/test_live.py
```

The live test starts a real controller, checks Docker runtime settings, and exercises SDK lifecycle, commands, file transfer, workspace restore, timeout cleanup, and tenant separation. CI also builds and starts the Compose controller. No live runtime test runs against a production deployment.

See [deployment](docs/deployment.md), [API compatibility](docs/compatibility.md), and [third-party notices](THIRD_PARTY_NOTICES.md).
