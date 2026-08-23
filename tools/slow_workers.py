"""A pytest plugin that delays every asyncio.to_thread call, to expose timing races.

The setup program's dialogs fill themselves from workers that leave the event loop.
The timezone picker reads tzdata that way, and the device picker probes the sound card.
A test that pauses the app once and then asserts is racing those workers: it wins on a
developer's machine and loses on a loaded runner.  Eleven tests in
tests/test_setup_app.py were in exactly that state, and CI showed only one of them,
once, after the tree had already merged.

Delaying to_thread makes the race lose every time, so a timing assumption fails here
rather than at some later hour on somebody else's branch.  Run it over any test that
drives a dialog with a worker behind it:

    PYTHONPATH=tools pytest tests/test_setup_app.py -p slow_workers --no-cov

Every test should pass with this loaded.  One that does not is asserting before the
work it depends on has finished, and the fix is to wait for the condition rather than
to sleep longer.  See _wait_until in tests/test_setup_app.py.

--no-cov because the gate is calibrated against the ordinary unit run, and this loads
a plugin that run does not.

The delay comes from BUZZ_SLOW_WORKER_S, defaulting to 0.3 seconds.  Anything well
past a single message-queue drain will do, and a larger value only makes the run
slower rather than the check stricter.
"""

import asyncio
import os
from collections.abc import Callable
from typing import Any

DEFAULT_DELAY_S = 0.3

# Read once, so every delayed call in a run waits the same amount and a failure is
# reproducible from the command that produced it.
_DELAY_VARIABLE = 'BUZZ_SLOW_WORKER_S'


def worker_delay_seconds(environment: dict[str, str] | None = None) -> float:
    """How long to hold each to_thread call, from the environment or the default.

    A bad value is ignored rather than refused.  This is a diagnostic aid, and failing
    the whole run over a typo in an optional variable would cost more than the delay it
    was trying to set.
    """
    raw = (environment if environment is not None else os.environ).get(_DELAY_VARIABLE)
    if raw is None:
        return DEFAULT_DELAY_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_DELAY_S


def make_slow_to_thread(real_to_thread: Callable[..., Any], delay_s: float) -> Callable[..., Any]:
    """Wrap asyncio.to_thread so it waits before handing the work to a thread.

    The wait happens on the event loop rather than in the thread, so it delays when the
    caller sees the result without changing what the called function does.
    """
    async def slow_to_thread(func, /, *args, **kwargs):
        await asyncio.sleep(delay_s)
        return await real_to_thread(func, *args, **kwargs)

    return slow_to_thread


def pytest_configure(config) -> None:  # pragma: no cover - pytest's own entry point
    asyncio.to_thread = make_slow_to_thread(asyncio.to_thread, worker_delay_seconds())
