"""REST lifecycle and interoperable command-session gateway."""

import asyncio
import base64
import fcntl
import hashlib
import hmac
import json
import logging
import os
import shlex
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from google.protobuf.message import DecodeError
from pydantic import BaseModel, ConfigDict, Field

from reliquary_sandbox._proto.command_session import command_session_pb2 as pb
from reliquary_sandbox.episode_task import VERIFIERS_ENTRY_PREFIX

from .manager import ACTIVE, CreateRequest, Manager, timestamp
from .settings import Settings

LOG = logging.getLogger(__name__)


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, max_length=65536)
    working_dir: str | None = None
    env: dict[str, str] | None = None
    timeout: float = Field(default=300, gt=0, le=86400, allow_inf_nan=False)
    user: str | None = None


class BatchRequest(BaseModel):
    sandbox_ids: list[str] = Field(max_length=100)


class CheckpointBatchRequest(BaseModel):
    checkpoint_ids: list[str] = Field(max_length=100)


class BulkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sandbox_ids: list[str] | None = Field(default=None, max_length=100)
    labels: list[str] | None = None
    all_users: bool = False
    team_id: str | None = None
    user_id: str | None = None


class BodyLimit:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0
        started = False

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                raise HTTPException(413, "request body exceeds configured limit")
            return message

        async def observed_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.limit:
                        return await JSONResponse({"detail": "request body exceeds configured limit"}, 413)(scope, receive, send)
                except ValueError:
                    return await JSONResponse({"detail": "invalid content length"}, 400)(scope, receive, send)
        try:
            await self.app(scope, bounded_receive, observed_send)
        except HTTPException as exc:
            if started:
                raise
            await JSONResponse({"detail": exc.detail}, exc.status_code)(scope, receive, send)


def _frame(payload, flag=0):
    return bytes([flag]) + struct.pack(">I", len(payload)) + payload


def _event(**kwargs):
    response = pb.StartResponse()
    if "stdout" in kwargs:
        response.event.data.stdout = kwargs["stdout"]
    elif "stderr" in kwargs:
        response.event.data.stderr = kwargs["stderr"]
    elif "exit_code" in kwargs:
        response.event.end.exit_code = kwargs["exit_code"]
    else:
        response.event.start.pid = kwargs["pid"]
    return _frame(response.SerializeToString())


def create_app(settings=None, runtime=None, task_runtime=None):
    settings = settings or Settings()
    stop = asyncio.Event()
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        settings.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_fd = os.open(settings.state_dir / "controller.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock_fd)
            raise RuntimeError("another controller owns this state directory") from None
        from .runtime import DockerRuntime

        bridged = sorted({entry for entry in settings.episode_envs.values()
                          if entry.startswith(VERIFIERS_ENTRY_PREFIX)})
        if settings.episodes_enabled and bridged:
            # Before any thread of ours: scoring processes fork from this server, never
            # from the service (see scoring_server), with the verifiers env modules
            # preloaded. Only verifiers entries are scored there: a config without one
            # starts no forkserver and the scoring path imports no other env's module.
            from . import scoring_server

            scoring_server.configure(wrapper=settings.scoring_wrapper,
                                     uid_base=settings.scoring_uid_base,
                                     slots=settings.scoring_slots,
                                     tmp_root=settings.scoring_tmp_root)
            from .episodes.registry import entry_module

            try:  # fail closed, before anything is imported or preloaded
                modules = sorted({
                    entry_module(entry,
                                 allow_unpackaged=settings.episode_allow_unpackaged_verifiers)
                    for entry in bridged})
            except Exception:
                os.close(lock_fd)
                raise
            try:
                scoring_server.start(preload=modules)
            except (OSError, ValueError) as exc:  # PermissionError: an unprivileged controller
                LOG.error("verifiers scoring is refused, every bridged grade will be aborted: "
                          "%s", exc)
            else:
                try:  # the forkserver skips a failed preload silently: refuse to start
                    scoring_server.check_preload(modules)
                except RuntimeError:
                    scoring_server.stop()
                    os.close(lock_fd)
                    raise
        manager = None
        reaper = None
        heartbeat = None
        host_checks = None
        # The reaper, host checks and capacity builds run on their own small pool;
        # episode routes on theirs. Neither can starve the other or the default executor.
        maintenance = ThreadPoolExecutor(2, thread_name_prefix="maintenance")
        episode_pool = None
        transcript_pool = None
        try:
            backend = runtime if runtime is not None else DockerRuntime(settings)
            backend.readiness()
            manager = Manager(settings, backend)
            manager.reconcile()
            state["manager"] = manager
            loop = asyncio.get_running_loop()
            if settings.episodes_enabled:
                # Recovery finishes before the episode routes serve and before the
                # reaper starts; until then they answer 503 "recovering".
                from .episodes.capacity import CapacityCache
                from .episodes.factory import HostChecks, build_episode_manager

                episodes = await loop.run_in_executor(
                    maintenance, build_episode_manager, settings, backend, manager.store,
                    task_runtime)
                await loop.run_in_executor(maintenance, episodes.reconcile)
                episode_pool = ThreadPoolExecutor(settings.episode_capacity * 2 + 4,
                                                  thread_name_prefix="episodes")
                # Transcript reads (up to episode_max_transcript_bytes each) on their own pool.
                transcript_pool = ThreadPoolExecutor(2, thread_name_prefix="transcripts")
                checks = HostChecks(episodes)
                state.update(episodes=episodes, episode_pool=episode_pool, host_checks=checks,
                             capacity=CapacityCache(episodes, maintenance),
                             transcript_pool=transcript_pool)
            stop.clear()

            async def periodic_host_checks(checks):
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(),
                                               timeout=settings.episode_host_check_seconds)
                    except TimeoutError:
                        pass
                    if not stop.is_set():
                        await loop.run_in_executor(maintenance, checks.check)

            async def cleanup():
                while not stop.is_set():
                    failed = False
                    try:
                        await loop.run_in_executor(maintenance, manager.reap)
                    except Exception:
                        failed = True
                    if "episodes" in state:
                        try:
                            await loop.run_in_executor(maintenance, state["episodes"].reap)
                        except Exception:
                            failed = True
                    if failed:
                        state["reaper_error"] = True
                    else:
                        state.pop("reaper_error", None)
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=settings.reaper_interval_seconds)
                    except TimeoutError:
                        pass

            reaper = asyncio.create_task(cleanup())
            if "episodes" in state:
                host_checks = asyncio.create_task(periodic_host_checks(state["host_checks"]))
            if "episodes" in state and settings.heartbeat_url:
                from .episodes import capacity

                heartbeat = asyncio.create_task(capacity.push_heartbeats(
                    settings.heartbeat_url, settings.heartbeat_interval_seconds,
                    state["capacity"].get, stop, state))
            yield
        finally:
            stop.set()
            for task in (reaper, heartbeat, host_checks):
                if task:
                    await task
            for name in ("episodes", "episode_pool", "host_checks", "capacity", "transcript_pool"):
                state.pop(name, None)
            for pool in (episode_pool, transcript_pool):
                if pool:
                    pool.shutdown(wait=True, cancel_futures=True)
            maintenance.shutdown(wait=True, cancel_futures=True)
            if manager:
                manager.store.close()
            os.close(lock_fd)

    app = FastAPI(title="Reliquary Sandbox", version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit, limit=settings.max_file_bytes + 1024**2)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"detail": "invalid request parameters"}, 422)

    def manager():
        return state["manager"]

    app.state.service = state  # read by the local gateway (conformance runs)

    if settings.episodes_enabled:
        from .episodes.routes import register_episode_routes

        register_episode_routes(app, state, settings)

    def tenant(authorization: Annotated[str | None, Header()] = None):
        supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        matched = None
        for name, token in settings.tokens.items():
            if hmac.compare_digest(supplied.encode(), token.encode()):
                matched = name
        if matched is None:
            raise HTTPException(401, "valid API key required", headers={"WWW-Authenticate": "Bearer"})
        return matched

    Tenant = Annotated[str, Depends(tenant)]

    def gateway(tenant_name, sandbox_id, authorization):
        token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        try:
            if len(token) > 4096:
                raise ValueError
            claims, signature = token.split(".", 1)
            secret = settings.tokens[tenant_name]
            expected = hmac.new(secret.encode(), claims.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            value = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))
            if not isinstance(value, dict) or value != {
                "tenant": tenant_name, "sandbox": sandbox_id, "exp": value.get("exp")
            } or type(value["exp"]) is not int or value["exp"] <= time.time():
                raise ValueError
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            raise HTTPException(401, "valid sandbox token required") from None
        manager().owned(tenant_name, sandbox_id, running=True)

    def command_options(command):
        if command.user not in (None, "", str(settings.default_uid), f"{settings.default_uid}:{settings.default_uid}"):
            raise HTTPException(422, "guest user override unavailable")
        if command.env and (len(command.env) > 64 or any(len(k) > 128 or len(v) > 8192 or "\x00" in k + v or "=" in k for k, v in command.env.items())):
            raise HTTPException(422, "invalid guest environment")

    @app.get("/health")
    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/readyz")
    def ready():
        try:
            manager().runtime.readiness()
            if settings.episodes_enabled and not state["host_checks"].ok:
                # The last periodic episode host check failed (absent while recovering).
                raise RuntimeError
            if state.get("reaper_error"):
                raise RuntimeError
            return {"status": "ready", "runtime": settings.docker_runtime}
        except Exception:
            return JSONResponse({"status": "unavailable"}, 503)

    @app.get("/api/v1/capabilities")
    def capabilities(owner: Tenant):
        return {"runtime": "gvisor" if settings.require_gvisor else "container",
                "lifecycle": True, "commands": ["connect-protobuf", "http-json"],
                "files": True, "workspace_checkpoints": True, "vm": False,
                "gpu": False, "network": "none", "ssh": False, "tunnels": False,
                "interactive_processes": False, "image_builds": False,
                "limits": {"file_bytes": settings.max_file_bytes, "checkpoint_bytes": settings.max_snapshot_bytes,
                           "output_bytes": settings.max_output_bytes, "command_seconds": settings.max_command_seconds}}

    @app.get("/api/v1/openapi.json")
    def openapi(owner: Tenant):
        return app.openapi()

    @app.get("/metrics")
    def metrics(owner: Tenant):
        rows = manager().store.all(owner)
        text = "# TYPE sandbox_count gauge\n" + "".join(f'sandbox_count{{status="{status}"}} {sum(r["status"] == status for r in rows)}\n' for status in sorted(ACTIVE | {"ERROR", "TERMINATED", "TIMEOUT"}))
        return Response(text, media_type="text/plain; version=0.0.4")

    @app.post("/api/v1/sandbox")
    def create(request: CreateRequest, owner: Tenant):
        if request.environment_vars and (len(request.environment_vars) > 64 or any(len(k) > 128 or len(v) > 8192 or "\x00" in k + v or "=" in k for k, v in request.environment_vars.items())):
            raise HTTPException(422, "invalid guest environment")
        return manager().create(owner, request)

    @app.get("/api/v1/sandbox")
    def listing(owner: Tenant, page: int = Query(1, ge=1), per_page: int = Query(50, ge=1, le=100),
                status: str | None = None, labels: Annotated[list[str] | None, Query()] = None,
                is_active: bool = False, team_id: str | None = None, user_id: str | None = None):
        if team_id or user_id:
            raise HTTPException(501, "external identity scopes unavailable")
        rows = [r for r in manager().store.all(owner) if (not status or r["status"] == status)
                and (not is_active or r["status"] in ACTIVE)
                and (not labels or set(labels).issubset(r["labels"]))]
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        start = (page - 1) * per_page
        return {"sandboxes": [manager().view(r) for r in rows[start:start + per_page]],
                "total": len(rows), "page": page, "perPage": per_page, "hasNext": start + per_page < len(rows)}

    @app.post("/api/v1/sandbox/status:batchGet")
    def batch(request: BatchRequest, owner: Tenant):
        statuses, errors = [], []
        for sandbox_id in request.sandbox_ids:
            try:
                row = manager().owned(owner, sandbox_id)
                statuses.append({"sandbox_id": sandbox_id, "status": row["status"],
                                 "error_type": row.get("error_type"), "error_message": row.get("error_message")})
            except HTTPException:
                errors.append({"sandbox_id": sandbox_id, "code": "NOT_FOUND", "message": "sandbox not found"})
        return {"statuses": statuses, "errors": errors}

    @app.post("/api/v1/sandbox/checkpoints:batchGet")
    def checkpoint_batch(request: CheckpointBatchRequest, owner: Tenant):
        results, errors = [], []
        for checkpoint_id in request.checkpoint_ids:
            item = manager().store.checkpoint(owner, checkpoint_id)
            if item:
                results.append(item[0])
            else:
                errors.append({"checkpoint_id": checkpoint_id, "code": "NOT_FOUND", "message": "checkpoint not found"})
        return {"checkpoints": results, "errors": errors}

    @app.get("/api/v1/sandbox/checkpoints/{checkpoint_id}")
    def get_checkpoint(checkpoint_id: str, owner: Tenant):
        item = manager().store.checkpoint(owner, checkpoint_id)
        if not item:
            raise HTTPException(404, "checkpoint not found")
        return item[0]

    @app.delete("/api/v1/sandbox/checkpoints/{checkpoint_id}")
    def delete_checkpoint(checkpoint_id: str, owner: Tenant):
        get_checkpoint(checkpoint_id, owner)
        manager().store.delete_checkpoint(owner, checkpoint_id)
        return {"success": True}

    @app.delete("/api/v1/sandbox")
    def bulk(request: BulkRequest, owner: Tenant):
        if request.all_users or request.team_id or request.user_id:
            raise HTTPException(501, "external identity scopes unavailable")
        if not request.sandbox_ids and not request.labels:
            raise HTTPException(422, "sandbox_ids or labels required")
        candidates = request.sandbox_ids if request.sandbox_ids else [r["id"] for r in manager().store.all(owner) if set(request.labels).issubset(r["labels"])]
        succeeded, failed = [], []
        for sandbox_id in candidates:
            try:
                manager().terminate(owner, sandbox_id)
                succeeded.append(sandbox_id)
            except HTTPException:
                failed.append({"sandbox_id": sandbox_id, "error": "sandbox unavailable"})
        return {"succeeded": succeeded, "failed": failed, "message": "bulk deletion complete"}

    @app.get("/api/v1/sandbox/{sandbox_id}")
    def get(sandbox_id: str, owner: Tenant):
        return manager().view(manager().owned(owner, sandbox_id))

    @app.delete("/api/v1/sandbox/{sandbox_id}")
    def delete(sandbox_id: str, owner: Tenant):
        return manager().terminate(owner, sandbox_id)

    @app.get("/api/v1/sandbox/{sandbox_id}/error-context")
    def error_context(sandbox_id: str, owner: Tenant):
        row = manager().owned(owner, sandbox_id)
        return {k: row.get(k) for k in ("status", "error_type", "error_message", "termination_reason", "exit_code")}

    @app.get("/api/v1/sandbox/{sandbox_id}/logs")
    def logs(sandbox_id: str, owner: Tenant):
        manager().owned(owner, sandbox_id)
        return {"logs": ""}

    @app.post("/api/v1/sandbox/{sandbox_id}/checkpoints")
    def checkpoint(sandbox_id: str, owner: Tenant):
        return manager().checkpoint(owner, sandbox_id)

    @app.get("/api/v1/sandbox/{sandbox_id}/checkpoints")
    def checkpoints(sandbox_id: str, owner: Tenant, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        manager().owned(owner, sandbox_id)
        rows = [r for r in manager().store.checkpoints(owner) if r["sandbox_id"] == sandbox_id]
        return {"checkpoints": rows[offset:offset + limit], "total": len(rows), "limit": limit, "offset": offset}

    @app.post("/api/v1/sandbox/{sandbox_id}/auth")
    def auth(sandbox_id: str, owner: Tenant):
        manager().owned(owner, sandbox_id, running=True)
        expiry = int(time.time()) + settings.gateway_token_seconds
        claims = base64.urlsafe_b64encode(json.dumps({"tenant": owner, "sandbox": sandbox_id, "exp": expiry}, sort_keys=True).encode()).decode().rstrip("=")
        signature = hmac.new(settings.tokens[owner].encode(), claims.encode(), hashlib.sha256).hexdigest()
        return {"gateway_url": settings.public_base_url + "/gateway", "user_ns": owner,
                "job_id": sandbox_id, "token": claims + "." + signature,
                "expires_at": datetime.fromtimestamp(expiry, timezone.utc).isoformat()}

    @app.post("/api/v1/sandbox/{sandbox_id}/command")
    def command(sandbox_id: str, request: CommandRequest, owner: Tenant):
        command_options(request)
        return manager().execute(owner, sandbox_id, request.command, request.timeout, request.working_dir, request.env)

    @app.post("/gateway/{tenant_name}/{sandbox_id}/execute")
    def gateway_command(tenant_name: str, sandbox_id: str, request: CommandRequest, authorization: Annotated[str | None, Header()] = None):
        gateway(tenant_name, sandbox_id, authorization)
        command_options(request)
        return manager().execute(tenant_name, sandbox_id, request.command, request.timeout, request.working_dir, request.env)

    @app.post("/gateway/{tenant_name}/{sandbox_id}/upload")
    async def upload(tenant_name: str, sandbox_id: str, request: Request, path: str, authorization: Annotated[str | None, Header()] = None):
        gateway(tenant_name, sandbox_id, authorization)
        async with request.form(max_files=1, max_fields=0, max_part_size=settings.max_file_bytes) as form:
            file = form.get("file")
            if not isinstance(file, UploadFile) and not hasattr(file, "read"):
                raise HTTPException(422, "multipart file required")
            data = await file.read(settings.max_file_bytes + 1)
            if len(data) > settings.max_file_bytes:
                raise HTTPException(413, "file exceeds configured limit")
            await asyncio.to_thread(manager().file, tenant_name, sandbox_id, path, data)
        return {"success": True, "path": path, "size": len(data), "timestamp": timestamp()}

    @app.get("/gateway/{tenant_name}/{sandbox_id}/download")
    def download(tenant_name: str, sandbox_id: str, path: str, authorization: Annotated[str | None, Header()] = None):
        gateway(tenant_name, sandbox_id, authorization)
        return Response(manager().file(tenant_name, sandbox_id, path), media_type="application/octet-stream")

    @app.get("/gateway/{tenant_name}/{sandbox_id}/read-file")
    def read_file(tenant_name: str, sandbox_id: str, path: str, offset: int = Query(0, ge=0), length: int | None = Query(None, ge=0), authorization: Annotated[str | None, Header()] = None):
        gateway(tenant_name, sandbox_id, authorization)
        data = manager().file(tenant_name, sandbox_id, path)
        window = data[offset:] if length is None else data[offset:offset + length]
        return {"content": window.decode("utf-8", "replace"), "size": len(window), "total_size": len(data), "offset": offset, "truncated": offset + len(window) < len(data)}

    @app.post("/gateway/{tenant_name}/{sandbox_id}/command_session.CommandSession/Start")
    async def rpc_start(tenant_name: str, sandbox_id: str, request: Request, authorization: Annotated[str | None, Header()] = None):
        gateway(tenant_name, sandbox_id, authorization)
        body = await request.body()
        try:
            if len(body) < 5 or body[0] != 0 or struct.unpack(">I", body[1:5])[0] != len(body) - 5:
                raise ValueError
            message = pb.StartRequest.FromString(body[5:])
            if message.stdin or message.session_uuid:
                return JSONResponse({"code": "unimplemented", "message": "interactive sessions unavailable"}, 501)
            command = CommandRequest(command=shlex.join([message.command.cmd, *message.command.args]),
                                     working_dir=message.command.cwd or None, env=dict(message.command.envs),
                                     timeout=min(float(request.headers.get("connect-timeout-ms", settings.max_command_seconds * 1000)) / 1000, settings.max_command_seconds),
                                     user=message.command.user or None)
            command_options(command)
        except (ValueError, DecodeError):
            return JSONResponse({"code": "invalid_argument", "message": "invalid command request"}, 400)

        async def events():
            try:
                result = await asyncio.to_thread(manager().execute, tenant_name, sandbox_id, command.command, command.timeout, command.working_dir, command.env)
                yield _event(pid=0)
                for stream in ("stdout", "stderr"):
                    data = result[stream].encode()
                    for start in range(0, len(data), 65536):
                        yield _event(**{stream: data[start:start + 65536]})
                if result.get("timed_out"):
                    yield _frame(json.dumps({"error": {"code": "deadline_exceeded", "message": "command timeout"}}).encode(), 2)
                elif result.get("truncated"):
                    yield _frame(json.dumps({"error": {"code": "resource_exhausted", "message": "command output limit exceeded"}}).encode(), 2)
                else:
                    yield _event(exit_code=result["exit_code"])
                    yield _frame(b"{}", 2)
            except HTTPException as exc:
                code = {404: "not_found", 409: "failed_precondition", 422: "invalid_argument"}.get(exc.status_code, "unavailable")
                yield _frame(json.dumps({"error": {"code": code, "message": exc.detail}}).encode(), 2)

        return StreamingResponse(events(), media_type="application/connect+proto", headers={"Connect-Protocol-Version": "1"})

    @app.api_route("/gateway/{tenant_name}/{sandbox_id}/command_session.CommandSession/{operation}", methods=["POST"])
    def unsupported_rpc(tenant_name: str, sandbox_id: str, operation: str, authorization: Annotated[str | None, Header()] = None):
        gateway(tenant_name, sandbox_id, authorization)
        return JSONResponse({"code": "unimplemented", "message": "interactive sessions unavailable"}, 501)

    @app.api_route("/api/v1/sandbox/{sandbox_id}/{operation:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unsupported(tenant_name: Tenant, sandbox_id: str, operation: str):
        manager().owned(tenant_name, sandbox_id)
        raise HTTPException(501, "capability unavailable on this runtime")

    @app.api_route("/api/v1/images/{operation:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def unsupported_image(owner: Tenant, operation: str):
        raise HTTPException(501, "image builds unavailable on this runtime")

    return app
