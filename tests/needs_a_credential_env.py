"""An env module that reads a credential of its environment at import time: the
service (which holds it) imports it, the scoring forkserver (empty environment) cannot."""

import os

from fake_episode_env import sandbox_task  # noqa: F401

CREDENTIAL = os.environ["RELIQUARY_TEST_IMPORT_CREDENTIAL"]
