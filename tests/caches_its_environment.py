"""A module that keeps a variable of its environment from import time (as a settings
object would): preloaded by the scoring forkserver, it is in every scoring process."""

import os

CACHED = os.environ.get("RELIQUARY_TEST_HEAP_SECRET", "")
