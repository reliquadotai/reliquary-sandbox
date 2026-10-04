"""Durable single-controller lifecycle state."""

import json
import sqlite3
import threading
from pathlib import Path


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        self.lock = threading.RLock()
        database = directory / "state.sqlite3"
        self.db = sqlite3.connect(database, check_same_thread=False)
        database.chmod(0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sandboxes (
                id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL,
                request_hash TEXT NOT NULL, idempotency_key TEXT,
                UNIQUE(tenant, idempotency_key)
            );
            CREATE INDEX IF NOT EXISTS sandbox_tenant ON sandboxes(tenant);
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL, snapshot BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS checkpoint_tenant ON checkpoints(tenant);
        """)
        self.db.commit()

    def save(self, record):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO sandboxes VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (record["id"], record["tenant"], json.dumps(record), record["request_hash"], record.get("idempotency_key")),
            )

    def get(self, sandbox_id):
        with self.lock:
            row = self.db.execute("SELECT payload FROM sandboxes WHERE id=?", (sandbox_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, tenant=None):
        with self.lock:
            rows = self.db.execute(
                "SELECT payload FROM sandboxes" + (" WHERE tenant=?" if tenant else ""),
                (tenant,) if tenant else (),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def idempotent(self, tenant, key):
        if not key:
            return None
        with self.lock:
            row = self.db.execute(
                "SELECT payload FROM sandboxes WHERE tenant=? AND idempotency_key=?", (tenant, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_checkpoint(self, tenant, payload, snapshot):
        with self.lock, self.db:
            self.db.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?)", (payload["id"], tenant, json.dumps(payload), snapshot))

    def checkpoint(self, tenant, checkpoint_id):
        with self.lock:
            row = self.db.execute("SELECT payload,snapshot FROM checkpoints WHERE id=? AND tenant=?", (checkpoint_id, tenant)).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def checkpoints(self, tenant):
        with self.lock:
            rows = self.db.execute("SELECT payload FROM checkpoints WHERE tenant=?", (tenant,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_checkpoint(self, tenant, checkpoint_id):
        with self.lock, self.db:
            self.db.execute("DELETE FROM checkpoints WHERE id=? AND tenant=?", (checkpoint_id, tenant))

    def prune_terminal(self, cutoff, maximum):
        """Bound completed history without discarding resources awaiting cleanup."""
        with self.lock, self.db:
            rows = self.db.execute("SELECT id,tenant,payload FROM sandboxes").fetchall()
            tenants = {}
            removed = []
            for sandbox_id, tenant, payload in rows:
                record = json.loads(payload)
                if record["status"] in {"PENDING", "PROVISIONING", "RUNNING"} or record.get("cleanup_pending"):
                    continue
                completed = record.get("terminated_at") or record["updated_at"]
                if completed < cutoff:
                    removed.append(sandbox_id)
                else:
                    tenants.setdefault(tenant, []).append((completed, sandbox_id))
            for records in tenants.values():
                records.sort(reverse=True)
                removed.extend(sandbox_id for _, sandbox_id in records[maximum:])
            self.db.executemany("DELETE FROM sandboxes WHERE id=?", ((sandbox_id,) for sandbox_id in removed))
            return removed

    def close(self):
        with self.lock:
            self.db.close()
