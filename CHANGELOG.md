# Changelog

## 0.1.0

- Added a self-hosted Linux container service for bounded miner workloads, agents, and evaluations, using approved Docker images with the gVisor runtime.
- Added tenant credentials, scoped guest gateway access, resource quotas, lifecycle reconciliation, lifetime limits, and command deadlines.
- Added synchronous and asynchronous Python SDKs with typed models, batch status polling, retry handling, Connect protobuf commands, and file transfer.
- Added bounded workspace file checkpoints and restore into new guests. Checkpoints retain filesystem contents only; process and VM state are outside this release.
- Added command-line lifecycle, command execution, file handling, checkpoint listing, restore, and checkpoint deletion workflows.
- Added pinned guest and controller builds, loopback Compose deployment, HTTPS proxy guidance, and a checksum-verified runtime installer.
- Added SDK, service, runtime, and CLI tests. Linux CI qualifies actual gVisor lifecycle, commands, file transfer, tenant separation, deadline cleanup, and workspace restore, then builds and starts the Compose controller.

Release qualification requires the unit checks, package build, Linux runtime integration suite, and Compose readiness check to pass for the revision being deployed. This release does not provide a hosted VM fleet, GPU allocation, SSH, public guest networking, image builds, or billing. Production deployment requires a configured Linux host, approved images, generated tenant credentials, and an HTTPS entry point for remote clients.
