# Contributing

Contributions are welcome — bug reports, feature ideas, and pull requests all appreciated.
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

Coverage must stay at or above 90 %.  The report will tell you which lines are
not yet exercised.

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
- Keep commit messages concise but informative — describe *why*, not just *what*.

## Reporting bugs

Please open a GitHub issue using the bug report template.  Include your operating
system, Python version, the relevant section of your config file (redact any
hostnames or credentials), and the log output around the failure.

## Feature requests

Open a GitHub issue using the feature request template.  The most useful requests
include a concrete use case — "I want to do X because Y" is much easier to act on
than "it would be cool if Z".
