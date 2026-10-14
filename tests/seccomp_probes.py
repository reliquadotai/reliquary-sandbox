"""Rewards that probe the scoring process's seccomp filter with raw syscalls (ctypes).

The audit hook refuses ctypes once a scoring process is ready, so these probes could not
reach the filter beneath it. This module's import (the task's unpickling, before `ready`:
import-time code of an env is trusted like its loading) turns the hook off in the
scoring processes that grade these tasks, and in no other: each scoring process is a
fresh fork of the forkserver, which never imports this module. The hook's own tests are
in test_bridge_grading ("the audit hook")."""

import json
import os

import verifiers.v1 as vf
from toy_bridge_env import ToyTask

import reliquary_sandbox_verifiers.scoring_process as scoring_process

scoring_process._install_audit_guard = lambda: None

NET_PROBE = "/tmp/net-probe.json"


class ProbesTheNetwork(ToyTask):
    """Scores 1 only when every non-Unix socket family and io_uring are refused with
    EPERM while Unix sockets work; writes what it saw to the box for the test."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import ctypes
        import socket

        seen = {}
        for name in ("AF_INET", "AF_INET6", "AF_NETLINK", "AF_PACKET", "AF_VSOCK"):
            try:
                socket.socket(getattr(socket, name), socket.SOCK_DGRAM).close()
                seen[name] = "opened"
            except PermissionError:
                seen[name] = "EPERM"
            except OSError as exc:
                seen[name] = f"errno {exc.errno}"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
                pass
            left, right = socket.socketpair()
            left.sendall(b"x")
            seen["AF_UNIX"] = "ok" if right.recv(1) == b"x" else "broken"
            left.close()
            right.close()
        except OSError as exc:
            seen["AF_UNIX"] = f"errno {exc.errno}"
        libc = ctypes.CDLL(None, use_errno=True)
        params = ctypes.create_string_buffer(120)  # struct io_uring_params
        fd = libc.syscall(425, 1, params)  # io_uring_setup: 425 on x86_64 and aarch64
        seen["io_uring"] = "EPERM" if fd < 0 and ctypes.get_errno() == 1 else f"{fd}"
        if fd >= 0:
            os.close(fd)
        with open("/proc/self/status") as status:
            seen["seccomp"] = [line.split()[1] for line in status
                               if line.startswith("Seccomp:")][0]
        await runtime.write(NET_PROBE, json.dumps(seen).encode())
        refused = all(seen[name] == "EPERM" for name in
                      ("AF_INET", "AF_INET6", "AF_NETLINK", "AF_PACKET", "AF_VSOCK"))
        return 1.0 if refused and seen["AF_UNIX"] == "ok" and seen["io_uring"] == "EPERM" else 0.0


class ProbesProcessReach(ToyTask):
    """Scores 1 only when ptrace, process_vm_readv/writev and pidfd_getfd fail with EPERM
    (with these arguments, unfiltered they fail with another errno or succeed)."""

    @vf.reward(weight=1.0)
    async def solved(self, runtime) -> float:
        import ctypes
        import platform

        numbers = {"x86_64": (101, 310, 311, 438),
                   "aarch64": (117, 270, 271, 438)}[platform.machine()]
        libc = ctypes.CDLL(None, use_errno=True)
        seen = {}
        for name, nr in zip(("ptrace", "process_vm_readv", "process_vm_writev",
                             "pidfd_getfd"), numbers):
            args = (0xFFFF,) if name == "ptrace" else (0, 0, 0)  # ptrace: invalid request
            result = libc.syscall(nr, *args)
            seen[name] = "EPERM" if result < 0 and ctypes.get_errno() == 1 else str(result)
        await runtime.write(NET_PROBE, json.dumps(seen).encode())
        return 1.0 if all(v == "EPERM" for v in seen.values()) else 0.0
