"""Command-line access to sandbox lifecycle and guest operations."""

import argparse
import json
import os
import shlex
import sys
from typing import Any

import httpx
from pydantic import BaseModel

from . import APIClient, CreateSandboxRequest, SandboxClient


def _emit(value: Any, as_json: bool) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    elif isinstance(value, list):
        value = [
            item.model_dump(mode="json", by_alias=True) if isinstance(item, BaseModel) else item
            for item in value
        ]
    if as_json or isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    elif value is not None:
        print(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="On-demand sandbox workspaces")
    parser.add_argument("--base-url", help="API URL (or RELIQUARY_SANDBOX_API_BASE_URL)")
    parser.add_argument("--api-key", help="API key (or RELIQUARY_SANDBOX_API_KEY)")
    parser.add_argument("--runtime", choices=("container", "vm"), default="container")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("create", help="Create an isolated workspace")
    create.add_argument("--name", default="eval-workspace")
    create.add_argument(
        "--image",
        default=os.getenv("RELIQUARY_SANDBOX_DEFAULT_IMAGE", "reliquary-sandbox-guest:0.1.0"),
    )
    create.add_argument("--cpu", type=float, default=1)
    create.add_argument("--memory", type=float, default=1, help="Memory limit in GB")
    create.add_argument("--disk", type=float, default=5, help="Workspace quota in GB")
    create.add_argument("--timeout", type=int, default=60, help="Lifetime in minutes")
    create.add_argument("--label", action="append", default=[])
    create.add_argument("--wait", action="store_true")
    listing = commands.add_parser("list", help="List accessible sandboxes")
    listing.add_argument("--status")
    listing.add_argument("--page", type=int, default=1)
    listing.add_argument("--per-page", type=int, default=50)
    commands.add_parser("get", help="Get a sandbox").add_argument("sandbox_id")
    run = commands.add_parser("run", help="Execute a guest command")
    run.add_argument("sandbox_id")
    run.add_argument("--cwd")
    run.add_argument("--timeout", type=int, default=300, help="Command timeout in seconds")
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("command", nargs="+")
    commands.add_parser("delete", help="Terminate and remove a sandbox").add_argument("sandbox_id")
    checkpoint = commands.add_parser("checkpoint", help="Snapshot the guest workspace files")
    checkpoint.add_argument("sandbox_id")
    checkpoint.add_argument("--wait", action="store_true", help="Wait for checkpoint durability")
    checkpoint.add_argument("--timeout", type=float, default=300, help="Durability wait in seconds")
    commands.add_parser("checkpoints", help="List a sandbox's workspace checkpoints").add_argument(
        "sandbox_id"
    )
    restore = commands.add_parser("restore", help="Create a container from a workspace checkpoint")
    restore.add_argument("checkpoint_id")
    restore.add_argument("--name", default="restored-workspace")
    restore.add_argument("--cpu", type=float, default=1)
    restore.add_argument("--memory", type=float, default=1, help="Memory limit in GB")
    restore.add_argument("--timeout", type=int, default=60, help="Lifetime in minutes")
    restore.add_argument("--wait", action="store_true")
    commands.add_parser("delete-checkpoint", help="Delete a retained workspace checkpoint").add_argument(
        "checkpoint_id"
    )
    upload = commands.add_parser("upload", help="Upload a local file")
    upload.add_argument("sandbox_id")
    upload.add_argument("local_path")
    upload.add_argument("remote_path")
    download = commands.add_parser("download", help="Download a guest file")
    download.add_argument("sandbox_id")
    download.add_argument("remote_path")
    download.add_argument("local_path")
    status = commands.add_parser("status", help="Get service health or sandbox status")
    status.add_argument("sandbox_id", nargs="?")
    for subcommand in commands.choices.values():
        subcommand.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.action == "restore" and args.runtime != "container":
        parser.error("restore requires the container runtime")
    if args.base_url:
        os.environ["RELIQUARY_SANDBOX_API_BASE_URL"] = args.base_url
    api = None
    try:
        api = APIClient(api_key=args.api_key)
        client = SandboxClient(api, runtime=args.runtime)
        if args.action == "create":
            sandbox = client.create(
                CreateSandboxRequest(
                    name=args.name,
                    docker_image=args.image,
                    cpu_cores=args.cpu,
                    memory_gb=args.memory,
                    disk_size_gb=args.disk,
                    timeout_minutes=args.timeout,
                    labels=args.label,
                )
            )
            if args.wait:
                client.wait_for_creation(sandbox.id)
                sandbox = client.get(sandbox.id)
            _emit(sandbox, args.json)
        elif args.action == "list":
            _emit(client.list(status=args.status, page=args.page, per_page=args.per_page), args.json)
        elif args.action in ("get", "status") and args.sandbox_id:
            _emit(client.get(args.sandbox_id), args.json)
        elif args.action == "status":
            response = httpx.get(f"{api.base_url}/health", timeout=10)
            response.raise_for_status()
            _emit(response.json(), args.json)
        elif args.action == "run":
            env = {}
            for item in args.env:
                if "=" not in item or not item.split("=", 1)[0]:
                    raise ValueError("--env must use KEY=VALUE")
                key, value = item.split("=", 1)
                env[key] = value
            command = args.command[0] if len(args.command) == 1 else shlex.join(args.command)
            result = client.execute_command(
                args.sandbox_id, command, working_dir=args.cwd, env=env, timeout=args.timeout
            )
            if args.json:
                _emit(result, True)
            else:
                sys.stdout.write(result.stdout)
                sys.stderr.write(result.stderr)
            return min(result.exit_code, 125) if result.exit_code >= 0 else 1
        elif args.action == "delete":
            _emit(client.delete(args.sandbox_id), args.json)
        elif args.action == "checkpoint":
            checkpoint = client.checkpoint(args.sandbox_id)
            if args.wait:
                checkpoint = client.wait_for_checkpoint(
                    checkpoint.id, timeout_seconds=args.timeout
                )
            _emit(checkpoint, args.json)
        elif args.action == "checkpoints":
            _emit(client.list_checkpoints(args.sandbox_id), args.json)
        elif args.action == "restore":
            sandbox = client.create(
                CreateSandboxRequest(
                    name=args.name,
                    checkpoint_id=args.checkpoint_id,
                    cpu_cores=args.cpu,
                    memory_gb=args.memory,
                    timeout_minutes=args.timeout,
                )
            )
            if args.wait:
                client.wait_for_creation(sandbox.id)
                sandbox = client.get(sandbox.id)
            _emit(sandbox, args.json)
        elif args.action == "delete-checkpoint":
            _emit(api.request("DELETE", f"/sandbox/checkpoints/{args.checkpoint_id}"), args.json)
        elif args.action == "upload":
            _emit(client.upload_file(args.sandbox_id, args.remote_path, args.local_path), args.json)
        elif args.action == "download":
            client.download_file(args.sandbox_id, args.remote_path, args.local_path)
            _emit({"path": args.local_path, "success": True}, args.json)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if args.json:
            print(json.dumps({"error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        else:
            print(f"Error: {error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if api is not None:
            api.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
