"""A drift pin over every patch target still written as a string.

`tests/patching.py` removes the risk for our own symbols by naming them with real
references.  What is left is the targets that stay strings on purpose - third-party
names reached through one of our modules, and the odd constant that has no `__name__`
to build a path from.  Those cannot break from a rename in `lib/`, but they can break
from an upgrade that moves something, and they break the same silent way: the test
that uses one fails, alone, minutes into a full run.

This resolves all of them in about a second, and names the ones that have gone.
"""

import importlib
import re
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

# patch('buzz.something.Name') and patch.object are the two forms in this suite.
# Only the first hides a name from every tool; patch.object already takes the class
# itself, and its attribute string fails loudly at the call rather than silently.
_TARGET = re.compile(r"""patch\(\s*['"](buzz\.[A-Za-z0-9_.]+)['"]""")


def _string_targets() -> set[str]:
    # This file is skipped because its own prose spells a target out as an example,
    # which the regex cannot tell from a real one.
    here = Path(__file__).resolve()
    return {match.group(1)
            for path in TESTS.rglob('*.py') if path.resolve() != here
            for match in _TARGET.finditer(path.read_text(encoding='utf-8'))}


def _resolves(target: str) -> bool:
    """Whether `target` names something that exists.

    The split between module and attributes is not knowable from the text - both
    `buzz.main.Collector` and `buzz.sampler.sd.InputStream` are a module path
    followed by attribute lookups, with the boundary in a different place.  So this
    walks from the longest importable prefix inwards, which is what `mock.patch`
    itself does.
    """
    parts = target.split('.')
    for split in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:split]))
        except Exception:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return False
        return True
    return False


def test_there_are_string_targets_left_to_check():
    """Guards the guard.  A regex that stopped matching would report every target
    resolvable and pass for ever, which is the failure mode a check that derives its
    verdict from an empty result always has.
    """
    assert len(_string_targets()) > 5


@pytest.mark.parametrize('target', sorted(_string_targets()))
def test_a_patch_target_still_names_something_that_exists(target):
    assert _resolves(target), (
        f'{target} does not resolve.  The test patching it fails at that call, with a '
        'message about the attribute rather than the rename that moved it.  For one '
        'of our own symbols, tests/patching.py builds the target from a real '
        'reference instead.')
