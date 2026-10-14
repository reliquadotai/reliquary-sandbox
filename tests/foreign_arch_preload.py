"""Preloaded in a test forkserver: the scoring processes forked from it believe they run
on an architecture the network filter does not know (riscv64), so they must refuse to
score."""

import reliquary_sandbox_verifiers.scoring_process as scoring_process

scoring_process.MACHINE = "riscv64"
