"""Patch targets built from real references, so that a rename cannot orphan one.

`mock.patch` takes the thing to replace as a string, and a string is invisible to
every tool that would otherwise catch a rename: an IDE will not update it, a grep for
the old name will not find it, and the type checker has nothing to check.  The failure
surfaces only when that one test runs, which in this suite means minutes later, in a
full run, after the work has already been called finished.  That has happened twice
here, both times to the same class being renamed in `lib/`.

Passing real references instead puts the names back in front of all three.

The module matters as much as the object.  `mock.patch` replaces a name where it is
looked up rather than where it is defined, so a test that wants `buzz.main` to build
something else has to patch the name bound in `buzz.main`, not the one in the module
that defines it.  That is why these take both halves rather than deriving the path
from the object alone, which would produce a target that patches the wrong binding
and a test that quietly proves nothing.

Two import shapes look alike in a patch string and do not behave alike.  Everything
above describes the first of them, where a module imports the symbol itself.

The second shape imports a module and reaches through it, as `buzz.recorder` does with
`from buzz import wavmeta` before it calls `wavmeta.append_metadata`.  The name bound
in `buzz.recorder` is the module, so there is only one binding of the function
anywhere.  `mock.patch` splits a target on its last dot and walks the rest, and a
prefix that passes through a module binding keeps going until it reaches whoever owns
the attribute.  `buzz.recorder.wavmeta.append_metadata` therefore patches
`buzz.wavmeta` itself, where `buzz.render` sees the same mock.  The prefix names a
route rather than a location, and it reads as a scope that it does not provide.

Name the owner for one of ours, because `patch_in(wavmeta, append_metadata)` says what
the string already did and claims no scope it lacks.  Reaching through to somebody
else's module goes further, since `patch('buzz.recorder.time.monotonic')` replaces the
clock for the whole process while the block runs.  Nothing depends on that today,
because `patch` restores cleanly and the unit suite runs serially in one process.  A
test that froze a clock around code which yields to another thread would find the
other thread sharing the frozen value.

Third-party names reached that way stay as strings.  They are not ours to rename, so
they carry none of the risk this exists to remove, and `buzz.sampler.sd.InputStream`
reads better than reaching for somebody else's module object.  The prefix is then the
only note of which of our modules the test cares about, which is the one job it still
does honestly.  `test_patch_targets.py` checks those still resolve.
"""

from types import ModuleType
from typing import Any
from unittest.mock import patch


def used_in(module: ModuleType, obj: Any) -> str:
    """The dotted path `mock.patch` needs for `obj` as `module` looks it up."""
    name = getattr(obj, '__name__', None)
    if name is None:
        raise TypeError(
            f'{obj!r} has no __name__, so a patch target cannot be built from it.  '
            'Name constants and instances as strings instead.  The module docstring '
            'says which targets are worth converting.')
    return f'{module.__name__}.{name}'


def patch_in(module: ModuleType, obj: Any, **kwargs: Any):
    """`mock.patch` for `obj` as `module` looks it up, with patch's own arguments."""
    return patch(used_in(module, obj), **kwargs)
