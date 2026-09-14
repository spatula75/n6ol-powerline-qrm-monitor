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

Third-party names reached through one of our modules - `buzz.sampler.sd.InputStream`
and the like - stay as strings.  They are not ours to rename, so they carry none of
the risk this exists to remove, and spelling them out reads better than reaching for
somebody else's module object.  `test_patch_targets.py` checks those still resolve.
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
