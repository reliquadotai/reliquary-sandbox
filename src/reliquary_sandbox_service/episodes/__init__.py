"""Signed episode gateway: open, call, finish, grade, sign.

Public here: `env_package_of` and `bridged_env_package_of`, the env package identities
validators allow-list."""

from .registry import bridged_env_package_of, env_package_of

__all__ = ["bridged_env_package_of", "env_package_of"]
