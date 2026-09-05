"""The platform gives an init budget before the clock starts (90 s per the rules page at
2026-09-05); every numba kernel compiles inside it. Half of it is the local bound, to
leave a factor of two for the platform's core."""

import subprocess
import sys
import time


def test_agent_imports_within_half_the_platform_budget() -> None:
    start = time.monotonic()
    subprocess.run([sys.executable, "-c", "import agent"], check=True, timeout=180)
    assert time.monotonic() - start < 45
