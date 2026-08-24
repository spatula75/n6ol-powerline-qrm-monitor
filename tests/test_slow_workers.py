"""Tests for tools/slow_workers.py, the plugin that exposes timing races.

The plugin's own job is to make other tests fail, so the thing worth pinning here is
that it delays without changing what the delayed call returns.  A wrapper that dropped
arguments or swallowed a result would turn every suite it was loaded over red, and the
cause would look like the suite rather than the tool.
"""

import asyncio
import time

from tools.slow_workers import DEFAULT_DELAY_S, make_slow_to_thread, worker_delay_seconds


class TestWorkerDelaySeconds:
    def test_it_defaults_when_the_variable_is_unset(self):
        assert worker_delay_seconds({}) == DEFAULT_DELAY_S

    def test_it_reads_the_variable(self):
        assert worker_delay_seconds({'BUZZ_SLOW_WORKER_S': '1.5'}) == 1.5

    def test_a_bad_value_falls_back_rather_than_failing_the_run(self):
        """A typo in an optional diagnostic variable must not cost the whole run."""
        assert worker_delay_seconds({'BUZZ_SLOW_WORKER_S': 'half a second'}) == DEFAULT_DELAY_S

    def test_a_negative_delay_becomes_no_delay(self):
        """asyncio.sleep refuses to go backwards, so clamp rather than raise."""
        assert worker_delay_seconds({'BUZZ_SLOW_WORKER_S': '-2'}) == 0.0


class TestMakeSlowToThread:
    def test_it_passes_arguments_through_and_returns_the_result(self):
        """The wrapper must be invisible apart from the wait.

        If it dropped an argument or lost the return value, every suite loaded with
        this plugin would fail somewhere unrelated, and the tool would look like the
        bug rather than the thing finding it.
        """
        async def scenario():
            slow = make_slow_to_thread(asyncio.to_thread, 0.0)
            return await slow(lambda a, b, c=0: (a, b, c), 1, 2, c=3)

        assert asyncio.run(scenario()) == (1, 2, 3)

    def test_it_actually_waits(self):
        async def scenario():
            slow = make_slow_to_thread(asyncio.to_thread, 0.2)
            started = time.monotonic()
            await slow(lambda: None)
            return time.monotonic() - started

        assert asyncio.run(scenario()) >= 0.2, (
            'The delay is the whole point.  Without it the plugin loads, changes '
            'nothing, and every racing test still passes.'
        )

    def test_an_exception_still_propagates(self):
        """A delayed call that fails has to fail the same way an undelayed one does."""
        def explode():
            raise ValueError('from the thread')

        async def scenario():
            slow = make_slow_to_thread(asyncio.to_thread, 0.0)
            await slow(explode)

        try:
            asyncio.run(scenario())
        except ValueError as exc:
            assert 'from the thread' in str(exc)
        else:
            raise AssertionError('The wrapper swallowed an exception from the thread.')
