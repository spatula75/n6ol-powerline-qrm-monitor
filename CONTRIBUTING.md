# Contributing

Contributions are welcome - bug reports, feature ideas, and pull requests all appreciated.
This project targets ham radio operators who want to document powerline interference, so
clarity and approachability matter as much as technical correctness.

## Setting up a development environment

```
git clone https://github.com/spatula75/n6ol-powerline-qrm-monitor.git
cd n6ol-powerline-qrm-monitor
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

## Running the tests

```
python -m pytest tests/
```

To also check coverage:

```
python -m pytest tests/ --cov
```

Coverage must stay at or above 97 %.  The report will tell you which lines are
not yet exercised.

### Integration tests

The suite above is the fast one, and it is what a plain `pytest` runs.  A second
suite under `tests/integration/` drives the real components - a real analyzer, a
real recorder, a real playback feeder - over real threads at real speed, and is
deselected by default because it costs about a minute:

```
python -m pytest -m integration --no-cov
```

`--no-cov` because the coverage gate is calibrated against what the unit suite
reaches; this run touches a different subset of the code and measuring it against
that number fails for no reason worth acting on.

They are slow *by construction* - playback is real-time because pacing is part of
what is under test - and so their assertions are properties ("locked within N
seconds", "the position advanced"), never exact timings.  Keep it that way: a test
that goes red on a loaded runner is a test everyone learns to ignore.  Some need
Qt, and render to its offscreen platform rather than a window, so they run on a
machine with no display and no sound card.

### Golden files

The DSP tests include golden files that lock in the exact numeric output of the
detection algorithm.  If you intentionally change the algorithm, regenerate them:

```
python tests/resources/generate_goldens.py
```

Commit the new `.npy` files alongside your code change.

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```
python -m ruff check .
```

All lint errors must be resolved before a pull request can be merged.  The
sequence for every change is: **make changes → run tests → run ruff → commit**.

Line length is 120 characters.  Type annotations are required on all public
functions and methods.

## Pull request guidelines

- One logical change per PR.
- Add or update tests for any changed behavior.
- Update `CHANGELOG.md` under `[Unreleased]` with a brief description of what changed.
- Keep commit messages concise but informative - describe *why*, not just *what*.

## Releasing

Deciding what to release is still done by hand; publishing it is not.

Do the following on a branch named `prepare-release-<version>` - `prepare-release-1.4.0`
for this release - so the branch name says which release it prepares without having
to open the PR to find out.

1. Bump the version in **both** `lib/buzz/__init__.py` and `pyproject.toml`.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new `## [x.y.z] - date`
   heading.  This section becomes the release notes verbatim, so write it for the
   person reading the release page.
3. Run the release render check against a recent recording:

   ```
   python tools/release_render_check.py
   ```

   This renders a real capture at both ends of the sample-rate band the monitor
   admits, and at several rates in between, then checks that each result actually
   holds a picture and a sound, not just a well-formed container - see the tool's
   own docstring for exactly what it checks.  It asks which recording to use when
   nothing recent is on hand rather than picking silently or skipping the check, so
   answer at the prompt instead of routing around it.  Not wired into CI: CI has no
   live radio and cannot produce a recording that means anything, so this stays a
   manual step for the same reason "Hands-on verification" in `CLAUDE.md` does.
4. Merge that as a PR, then tag the merge commit and push the tag:

   ```
   git tag 1.3.0 && git push origin 1.3.0
   ```

Pushing the tag runs `.github/workflows/release.yml`, which checks that the two
version strings and the tag agree and that the changelog has a section for it, runs
ruff, the unit suite under both the interpreted and compiled Numba paths, and the
integration suite - and only then builds the `.tar.gz` and `.zip` archives, a
`SHA256SUMS` beside them, and publishes the GitHub release.

If the verification fails, the tag exists but nothing has been published.  Delete
it (`git push --delete origin 1.3.0`), fix the problem, and tag again.

## Reporting bugs

Please open a GitHub issue using the bug report template.  Include your operating
system, Python version, the relevant section of your config file (redact any
hostnames or credentials), and the log output around the failure.

## Feature requests

Open a GitHub issue using the feature request template.  The most useful requests
include a concrete use case - "I want to do X because Y" is much easier to act on
than "it would be cool if Z".
