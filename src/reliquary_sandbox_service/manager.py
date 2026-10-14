"""Bounded sandbox lifecycle and tenant ownership."""

import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from pydantic import ConfigDict, Field

from reliquary_sandbox import CreateSandboxRequest

from .store import Store

ACTIVE = {"PENDING", "PROVISIONING", "RUNNING"}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class CreateRequest(CreateSandboxRequest):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=128)
    vm: bool = False
    cpu_cores: float = Field(default=1, gt=0, le=64, allow_inf_nan=False)
    memory_gb: float = Field(default=1, ge=0.125, le=64, allow_inf_nan=False)
    disk_size_gb: float | None = Field(default=None, ge=0.01, le=128, allow_inf_nan=False)
    timeout_minutes: int = Field(default=60, ge=1)
    labels: list[str] = Field(default_factory=list, max_length=32)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class Manager:
    def __init__(self, settings, runtime):
        self.settings = settings
        self.runtime = runtime
        self.store = Store(settings.state_dir)
        self.lock = threading.RLock()
        self.operation_locks: dict[str, threading.RLock] = {}
        self.checkpoint_locks = {tenant: threading.Lock() for tenant in settings.tokens}

    def operation_lock(self, sandbox_id):
        with self.lock:
            if self.store.get(sandbox_id) is None:
                raise HTTPException(404, "sandbox not found")
            return self.operation_locks.setdefault(sandbox_id, threading.RLock())

    @contextmanager
    def operation(self, tenant, sandbox_id):
        with self.lock:
            self.owned(tenant, sandbox_id)
            operation = self.operation_lock(sandbox_id)
            if not operation.acquire(blocking=False):
                raise HTTPException(409, "sandbox is busy")
        try:
            yield
        finally:
            operation.release()

    def owned(self, tenant, sandbox_id, running=False):
        with self.lock:
            record = self.store.get(sandbox_id)
            if not record or record["tenant"] != tenant:
                raise HTTPException(404, "sandbox not found")
            if running:
                if record["status"] != "RUNNING" or record.get("cleanup_pending"):
                    raise HTTPException(409, "sandbox is not running")
                now = time.time()
                expired = now >= record["expires_at"]
                idle = record.get("idle_timeout_minutes")
                operation = self.operation_locks.get(sandbox_id)
                acquired = operation.acquire(blocking=False) if operation else True
                try:
                    idle_expired = acquired and idle and now - record["last_activity"] >= idle * 60
                    if expired or idle_expired:
                        self.terminate(tenant, sandbox_id, "TIMEOUT", "lifetime expired" if expired else "idle timeout")
                        raise HTTPException(409, "sandbox lifetime or idle deadline expired")
                finally:
                    if operation and acquired:
                        operation.release()
            return record

    def prune_history(self):
        with self.lock:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=self.settings.retention_days)).isoformat()
            self.store.prune_terminal(cutoff, self.settings.max_terminal_records_per_tenant)
            known = {record["id"] for record in self.store.all()}
            for sandbox_id, operation in list(self.operation_locks.items()):
                if sandbox_id not in known and operation.acquire(blocking=False):
                    operation.release()
                    del self.operation_locks[sandbox_id]

    @staticmethod
    def view(record):
        return {key: value for key, value in record.items() if key not in {
            "tenant", "request_hash", "idempotency_key", "container_id", "last_activity",
            "expires_at", "request", "parent_checkpoint_id",
            "cleanup_pending",
            "cleanup_status", "cleanup_reason",
        }}

    def create(self, tenant, request):
        payload = request.model_dump(mode="json", exclude_none=True)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self.lock:
            self.prune_history()
            previous = self.store.idempotent(tenant, request.idempotency_key)
            if previous:
                if previous["request_hash"] != digest:
                    raise HTTPException(409, "idempotency key was used with a different request")
                return self.view(previous)
            unsupported = {
                "vm": request.vm, "gpu": request.gpu_count, "secrets": request.secrets,
                "egress": request.network_allowlist is not None or request.network_denylist is not None,
                "start_command": request.start_command, "advanced_configs": request.advanced_configs,
                "team_id": request.team_id, "region": request.region,
            }
            if any(unsupported.values()):
                raise HTTPException(501, "requested capability is unavailable on this runtime")
            if any(len(label) > 128 for label in request.labels):
                raise HTTPException(422, "labels must be at most 128 characters")
            if request.timeout_minutes > self.settings.max_timeout_minutes:
                raise HTTPException(422, "sandbox lifetime exceeds the configured limit")
            snapshot = None
            checkpoint = None
            if request.checkpoint_id:
                checkpoint = self.store.checkpoint(tenant, request.checkpoint_id)
                if not checkpoint:
                    raise HTTPException(404, "checkpoint not found")
                payload["docker_image"] = checkpoint[0]["docker_image"]
                payload["disk_size_gb"] = checkpoint[0]["disk_size_bytes"] / 1024**3
                snapshot = checkpoint[1]
            payload.setdefault("docker_image", self.settings.default_image)
            payload.setdefault("disk_size_gb", 5)
            active = [r for r in self.store.all(tenant) if r["status"] in ACTIVE or r.get("cleanup_pending")]
            if (len(active) >= self.settings.max_sandboxes_per_tenant
                or sum(r["cpu_cores"] for r in active) + request.cpu_cores > self.settings.max_cpu_per_tenant
                or sum(r["memory_gb"] for r in active) + request.memory_gb > self.settings.max_memory_gb_per_tenant):
                raise HTTPException(429, "tenant resource quota exceeded", headers={"Retry-After": "5"})
            sandbox_id = str(uuid.uuid4())
            now = timestamp()
            record = {
                "id": sandbox_id, "tenant": tenant, "name": request.name,
                "docker_image": payload["docker_image"], "cpu_cores": request.cpu_cores,
                "memory_gb": request.memory_gb, "disk_size_gb": payload["disk_size_gb"],
                "disk_mount_path": "/workspace", "gpu_count": 0, "vm": False,
                "status": "PROVISIONING", "timeout_minutes": request.timeout_minutes,
                "idle_timeout_minutes": request.idle_timeout_minutes,
                "labels": request.labels, "created_at": now, "updated_at": now,
                "request_hash": digest, "idempotency_key": request.idempotency_key,
                "last_activity": time.time(), "expires_at": time.time() + request.timeout_minutes * 60,
                "request": payload, "parent_checkpoint_id": request.checkpoint_id,
            }
            self.store.save(record)
            try:
                record["container_id"] = self.runtime.create(sandbox_id, payload)
                if snapshot is not None:
                    self.runtime.import_workspace(sandbox_id, snapshot)
                record.update(status="RUNNING", started_at=timestamp(), updated_at=timestamp())
                self.store.save(record)
            except Exception as exc:
                cleanup_pending = False
                try:
                    self.runtime.delete(sandbox_id)
                except Exception:
                    cleanup_pending = True
                record.update(status="ERROR", cleanup_pending=cleanup_pending, error_type="PROVISIONING_FAILED", error_message="runtime provisioning failed", updated_at=timestamp())
                self.store.save(record)
                self.prune_history()
                if isinstance(exc, ValueError):
                    raise HTTPException(422, "request exceeds runtime limits or uses an unavailable image") from exc
                raise HTTPException(503, "sandbox runtime unavailable") from exc
            return self.view(record)

    def terminate(self, tenant, sandbox_id, status="TERMINATED", reason="deleted"):
        # Deletion can interrupt an execution; the runtime removes all guest processes.
        with self.lock:
            record = self.owned(tenant, sandbox_id)
            if record["status"] in ACTIVE or record["status"] == "ERROR" or record.get("cleanup_pending"):
                try:
                    self.runtime.delete(sandbox_id)
                except Exception as exc:
                    record.update(status="ERROR", cleanup_pending=True, cleanup_status=status,
                                  cleanup_reason=reason, error_type="CLEANUP_FAILED",
                                  error_message="runtime cleanup pending", updated_at=timestamp())
                    self.store.save(record)
                    raise HTTPException(503, "runtime cleanup unavailable") from exc
                record.update(status=status, cleanup_pending=False, termination_reason=reason, terminated_at=timestamp(), updated_at=timestamp())
                record.pop("cleanup_status", None)
                record.pop("cleanup_reason", None)
                self.store.save(record)
                self.prune_history()
            return {"success": True, "id": sandbox_id}

    def execute(self, tenant, sandbox_id, command, timeout, cwd=None, env=None):
        with self.operation(tenant, sandbox_id):
            with self.lock:
                record = self.owned(tenant, sandbox_id, running=True)
                remaining = record["expires_at"] - time.time()
                if remaining <= 0:
                    self.terminate(tenant, sandbox_id, "TIMEOUT", "lifetime expired")
                    raise HTTPException(409, "sandbox lifetime expired")
                effective = min(timeout, self.settings.max_command_seconds, remaining)
                record["last_activity"] = time.time()
                self.store.save(record)
            try:
                result = self.runtime.execute(sandbox_id, command, effective, cwd=cwd, env=env)
            except ValueError as exc:
                raise HTTPException(422, "invalid guest command options") from exc
            except Exception as exc:
                self._failed_operation(tenant, sandbox_id)
                raise HTTPException(503, "guest command unavailable") from exc
            with self.lock:
                record = self.owned(tenant, sandbox_id)
                record["last_activity"] = time.time()
                record["updated_at"] = timestamp()
                if result.get("timed_out"):
                    record.update(status="TIMEOUT", termination_reason="command timeout", terminated_at=timestamp())
                self.store.save(record)
                if result.get("timed_out"):
                    self.prune_history()
            return result

    def file(self, tenant, sandbox_id, path, data=None):
        with self.operation(tenant, sandbox_id):
            self.owned(tenant, sandbox_id, running=True)
            try:
                result = self.runtime.download(sandbox_id, path) if data is None else self.runtime.upload(sandbox_id, path, data)
            except FileNotFoundError as exc:
                raise HTTPException(404, "file not found") from exc
            except ValueError as exc:
                raise HTTPException(422, "invalid file path or size") from exc
            except Exception as exc:
                self._failed_operation(tenant, sandbox_id)
                raise HTTPException(503, "guest filesystem unavailable") from exc
            with self.lock:
                record = self.owned(tenant, sandbox_id)
                record["last_activity"] = time.time()
                self.store.save(record)
            return result

    def checkpoint(self, tenant, sandbox_id):
        with self.operation(tenant, sandbox_id):
            admission = self.checkpoint_locks[tenant]
            if not admission.acquire(blocking=False):
                raise HTTPException(409, "checkpoint creation is busy")
            try:
                with self.lock:
                    record = self.owned(tenant, sandbox_id, running=True)
                    if len(self.store.checkpoints(tenant)) >= self.settings.max_checkpoints_per_tenant:
                        raise HTTPException(429, "checkpoint quota exceeded")
                    record["last_activity"] = time.time()
                    self.store.save(record)
                try:
                    snapshot = self.runtime.export_workspace(sandbox_id)
                except ValueError as exc:
                    raise HTTPException(422, "workspace exceeds checkpoint limits") from exc
                except Exception as exc:
                    self._failed_operation(tenant, sandbox_id)
                    raise HTTPException(503, "guest checkpoint unavailable") from exc
                if len(snapshot) > self.settings.max_snapshot_bytes:
                    raise HTTPException(422, "workspace exceeds checkpoint limits")
                with self.lock:
                    record = self.owned(tenant, sandbox_id, running=True)
                    now = timestamp()
                    parent = record.get("parent_checkpoint_id")
                    prior = self.store.checkpoint(tenant, parent) if parent else None
                    payload = {
                        "id": str(uuid.uuid4()), "sandbox_id": sandbox_id, "parent_id": parent,
                        "state": "DURABLE", "depth": prior[0]["depth"] + 1 if prior else 0,
                        "docker_image": record["docker_image"], "disk_size_bytes": int(record["disk_size_gb"] * 1024**3),
                        "stored_bytes": len(snapshot), "created_at": now, "updated_at": now,
                    }
                    self.store.save_checkpoint(tenant, payload, snapshot)
                    record["last_activity"] = time.time()
                    self.store.save(record)
                    return payload
            finally:
                admission.release()

    def _failed_operation(self, tenant, sandbox_id):
        with self.lock:
            record = self.owned(tenant, sandbox_id)
            if record["status"] not in ACTIVE:
                return
            record.update(status="ERROR", cleanup_pending=True, error_type="RUNTIME_OPERATION_FAILED", error_message="runtime operation failed", updated_at=timestamp())
            self.store.save(record)
            try:
                self.terminate(tenant, sandbox_id, "ERROR", "runtime operation failed")
            except HTTPException:
                pass

    def reconcile(self):
        with self.lock:
            managed = self.runtime.managed_sandboxes()
            known = {r["id"]: r for r in self.store.all()}
            for sandbox_id in managed:
                if sandbox_id not in known or known[sandbox_id]["status"] != "RUNNING":
                    self.runtime.delete(sandbox_id)
            for record in known.values():
                if record["status"] in ACTIVE and (record["status"] != "RUNNING" or self.runtime.status(record["id"]) != "running"):
                    self.runtime.delete(record["id"])
                    record.update(status="ERROR", error_type="RUNTIME_STOPPED", error_message="runtime stopped", updated_at=timestamp(), terminated_at=timestamp())
                    self.store.save(record)
                elif record.get("cleanup_pending"):
                    self.terminate(record["tenant"], record["id"], record.get("cleanup_status", record["status"]), record.get("cleanup_reason", "runtime cleanup"))
            self.prune_history()

    def reap(self):
        with self.lock:
            now = time.time()
            failure = None
            for record in self.store.all():
                if record.get("cleanup_pending"):
                    try:
                        self.terminate(record["tenant"], record["id"], record.get("cleanup_status", record["status"]), record.get("cleanup_reason", "runtime cleanup"))
                    except HTTPException as exc:
                        failure = exc
                    continue
                if record["status"] not in ACTIVE:
                    continue
                idle = record.get("idle_timeout_minutes")
                # Busy requests renew activity on completion; lifetime remains a hard deadline.
                lock = self.operation_lock(record["id"])
                acquired = lock.acquire(blocking=False)
                try:
                    expired = now >= record["expires_at"]
                    idle_expired = acquired and idle and now - record["last_activity"] >= idle * 60
                    if expired or idle_expired:
                        self.terminate(record["tenant"], record["id"], "TIMEOUT", "lifetime expired" if expired else "idle timeout")
                except HTTPException as exc:
                    failure = exc
                finally:
                    if acquired:
                        lock.release()
            self.prune_history()
            if failure:
                raise RuntimeError("runtime cleanup incomplete") from failure
