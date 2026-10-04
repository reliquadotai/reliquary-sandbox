# API compatibility

The Python package is `reliquary-sandbox`, imported as `reliquary_sandbox`. Its typed SDK surface includes lifecycle and checkpoint operations, file handling, command transport, image management clients, process clients, retries, and batch polling. Select a compatible API with `RELIQUARY_SANDBOX_API_BASE_URL` and its credential with `RELIQUARY_SANDBOX_API_KEY`.

The self-hosted service implements a bounded container runtime. `SandboxClient` and `AsyncSandboxClient` default to `runtime="container"`. Use `runtime="vm"` only with an API that actually provides VM-backed sandboxes; this service rejects VM requests.

## Supported protocol

Lifecycle requests preserve the SDK JSON models, field aliases, sandbox statuses, checkpoint states, and batch lookup response shapes. File operations use the SDK's authenticated gateway paths. Commands use the binary Connect RPC `command_session.CommandSession/Start` method and the generated protobuf event models. Both synchronous and asynchronous SDK execution use this command transport.

Workspace checkpoints preserve files only. `DURABLE` means the bounded archive is persisted in the controller's state directory; it does not mean a process or VM checkpoint exists. Restore requires the state volume and the original approved guest image to remain available.

Authenticated `GET /api/v1/capabilities` reports the runtime's available operations and configured bounds. Authenticated `GET /api/v1/openapi.json` returns the current REST schema. `/health` checks the controller process and `/readyz` checks the configured runtime.

## Explicit limits

There is no VM fleet, GPU allocator, SSH access service, guest public network access, registry publishing, image build service, billing service, or reconnectable background process session service in this deployment. The related SDK models and clients are retained for integrations targeting compatible APIs, but this controller returns explicit unsupported-operation responses.

Do not infer hosted-service parity from a shared request format. Integration qualification should exercise the exact endpoints and workload assumptions used by the caller.

## Evaluation integration

Create one sandbox per evaluation or bounded workload, upload the necessary input files to `/workspace`, execute the evaluation command with an explicit timeout, and download the resulting artifacts before deleting the sandbox. Preinstall dependencies in an approved guest image because guests cannot fetch packages from the network.

Use `environment_vars` for workload configuration. The SDK retains the `secrets` model for compatible providers, but this controller does not provide managed secret injection. A command can read its own guest environment; printed guest values become output and should be handled accordingly. Tenant credentials belong on the client/controller side, outside guest environments.
