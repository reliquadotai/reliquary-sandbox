"""A task module whose import takes SLOW_IMPORT_S (an env loading a large model or
dataset when it is imported), for the scoring process's ready limit and preload tests."""

import time

from toy_bridge_env import ToyTask

SLOW_IMPORT_S = 3.0
time.sleep(SLOW_IMPORT_S)


class SlowImportTask(ToyTask):
    pass
