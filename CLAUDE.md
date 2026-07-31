# CLAUDE.md

Working notes for Claude Code on this project. Distilled from prior sessions.

## What this is

A continuous powerline QRM monitor for ham radio operators. It samples receiver audio,
locks onto the 120 pps (60 Hz grid; 100 pps where mains is 50 Hz) impulse train from
arcing hardware, and logs signal and noise-floor levels to CSV, charts, and optional
`.wav` event recordings. There is a live Qt display: waterfall, phase-synchronised
oscilloscope, and S-unit bar graphs.

It is a **field tool, not lab test equipment**. The goal is knowing when PG&E is making
a racket, not chasing the last decibel. Precision work that costs clarity or testability
is usually not worth it — say so rather than building it.

## Layout

- `lib/buzz/` — the application. One responsibility per module.
- `tools/` — standalone diagnostic scripts.
- `tests/` — unit tests. `tests/integration/` is markered and deselected by default.
- `configure.py`, `level_meter.py` — end-user setup helpers at the root.
- `docs/`, `README.md`, `README-analysis.md` — user- and design-facing docs.
- `templates/index.html` — Jinja2 template for the published web page.
- `config.example.toml` — the documented sample config. Real config lives at
  `~/.buzz/config.toml`; the app falls back to dataclass defaults when it's absent.
- `tmp/` — gitignored scratch space. Notes and bug reports live here, uncommitted.

## Environment

Python 3.12+ (`requires-python = ">=3.12"`). CI pins 3.12; the local dev venv is
`.venv314` on Python 3.14. `lib/` is not installed as a package — it has to be on the
import path. `tests/conftest.py` inserts it for the suite; for a manual run set
`PYTHONPATH=lib` (PyCharm run configurations have `lib` as a source root instead).

Runtime deps are pinned in `requirements.txt`; `pyproject.toml` carries the looser
ranges for packaging. `numpy` is capped below 2.5 by numba's own constraint — check
that ceiling before bumping either.

## Commands

    pytest --cov                      # unit suite; must stay at or above the 97% gate
    pytest -m integration --no-cov    # integration tier; slow, real threads and time
    NUMBA_DISABLE_JIT=0 pytest tests/ # the compiled path production actually runs
    ruff check .                      # must be clean

    PYTHONPATH=lib python -m buzz.main            # run the monitor
    PYTHONPATH=lib python -m buzz.main --headless # no GUI
    python configure.py                            # pick the audio input device
    python level_meter.py                          # live S-meter for setting RF/AF gain

`main.py` flags: `--headless`, `--top`, `--enable-recording`, `--playback FILE`,
`--mute`, `--playback-gain DB`. Playback replays a recorded `.wav` through the whole
pipeline and suppresses CSV, plots, uploads, and recording.

The coverage gate lives in `pyproject.toml` as the single source of truth — don't pass
`--cov-fail-under` on the command line, it silently overrides it.

## CI and release workflows

Both live in `.github/workflows/` and share the `.github/actions/setup` composite
action, deliberately: a release is verified in the same environment every PR is tested
in, so the two cannot drift apart.

`ci.yml` runs on push and PR to `main`, as two parallel jobs:

- **test** — ruff, then `pytest tests/ --cov`, then the whole suite again with
  `NUMBA_DISABLE_JIT=0`. That second run is not redundant: the interpreted and JIT'd
  paths are not automatically equivalent — NumPy's NEP 50 promotion applies to mixed
  float32/float64 arithmetic in the interpreter while Numba types the expression
  itself, and `average_pulse_amplitude` returned measurably different answers under the
  two before an explicit cast fixed it.
- **integration** — a separate job rather than a later step, so a PR doesn't wait for
  both end to end.

`release.yml` runs on a pushed semver tag (or `workflow_dispatch` with a tag), also as
two jobs:

- **verify** — the gate. Confirms the tag, `lib/buzz/__init__.py`, and `pyproject.toml`
  all state the same version, and that `CHANGELOG.md` has a `## [x.y.z]` section; then
  everything CI runs, plus the integration suite.
- **publish** — `needs: verify`, so no archive exists until the gate passes. Builds
  `.tar.gz` and `.zip` from `git archive` against the tag with a version-stamped prefix,
  writes `SHA256SUMS` beside them, lifts that version's changelog section as the release
  notes, and publishes with `gh release create --verify-tag`.

`git archive` ships exactly what is committed at the tag — no build artefacts, no stray
working-copy files, and `.gitignore`d paths cannot leak in. These archives exist
alongside GitHub's own generated "Source code" ones because GitHub builds those on
demand and has changed their compression before now, silently altering published
checksums; these are built once and stay true.

Headless Linux needs `libportaudio2` for sounddevice and `libegl1 libgl1 libxkbcommon0`
for PySide6 — QtGui links against those at *import* time, so even pure-function tests
that never create a `QApplication` need them.

## Architecture

Threads and a ring buffer, not a call chain. Understand this before changing anything
in the audio or analysis path:

1. **Audio thread** — reads the input device continuously into a ring buffer and
   publishes that data is available. Nothing else touches the device.
2. **Analysis thread** (`ContinuousAnalyzer`) — reads the buffer at the offsets it
   wants, locks onto the pulse train, tracks phase drift by least squares, and
   publishes state transitions and results to registered listeners.
3. **Consumers** — the Qt display (waterfall, scope, bar graphs), the event recorder,
   and the once-a-minute collector all subscribe. The collector averages the last
   minute of published analysis rather than sampling anything itself.

Consequences worth remembering: analysis always looks slightly into the past, which is
fine for this measurement; consumers never wait on fresh audio because the buffer
already holds what they need; and a listener must not block, because it runs on the
publisher's thread.

`--playback` swaps the audio thread's source for a `.wav` and can use the output device
as the clock; muting switches clock sources without losing position.

## Git workflow

**Always ask before `git add`, `commit`, or `push`.** Never commit on your own
initiative, not even when the work is obviously finished and tests pass.

- Feature work goes on a branch, then a PR, then squash-merge to `main`.
- Commit granularity varies — sometimes one commit per unit of work, sometimes a single
  sweep for a batch of PR fixes. Ask which is wanted.
- Note `CHANGELOG.md` under `[Unreleased]` as part of the work, not at release time.

### Releasing

`CONTRIBUTING.md` has the procedure for humans; this is what not to get wrong.

1. **Bump the version in both `lib/buzz/__init__.py` and `pyproject.toml`.** Two files,
   easy to half-do, and the release workflow fails the release if they and the tag
   disagree.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new `## [x.y.z] — date`
   heading. That section is lifted verbatim as the release notes, so write it for
   whoever lands on the release page.
3. Merge as a PR, then tag the merge commit and push the tag: `git tag 1.3.0 &&
   git push origin 1.3.0`. Plain semver, **no leading `v`** — the workflow's tag filter
   won't match one.

Pushing the tag is what publishes. **The tag goes up before the tests run**, so a failed
verification leaves a tag with nothing attached — which is the intended failure mode, not
a bug. Recover by deleting the tag (`git push --delete origin 1.3.0`), fixing, and
tagging again.

## Before every commit

Tests first, then lint, then human eyes. In that order:

1. **Unit suite with coverage** — `pytest --cov`. Must pass, and coverage must stay at
   or above the 97% gate. Running plain `pytest` without `--cov` hides a coverage
   failure that CI then catches; that has broken the build before.
2. **Lint** — `ruff check .`. Must come back completely clean, not merely free of *new*
   complaints. Fix what it reports rather than suppressing it.
3. **Integration tier**, when the change touches audio, analysis, playback, recording,
   or the display — `pytest -m integration --no-cov`. CI runs it on every PR, so
   catching a failure locally is cheaper than catching it on GitHub.
4. **Hands-on verification.** Changes to the audio path, the display, or the charts get
   checked against a live radio before being committed — green tests are not the finish
   line for those.

Then ask before staging anything; see Git workflow.

A locally green run still isn't the whole story: CI also runs the entire suite with
`NUMBA_DISABLE_JIT=0` against the compiled Numba path, which does not always agree with
the interpreted one.

### Lint rules

- **120 columns.** Set as `line-length` under `[tool.ruff]` in `pyproject.toml`. Not 80
  — that limit chops up otherwise readable lines for no benefit on a modern screen.
- Ruff is configured for `E`, `F`, `W`, and `I`. `I` is import sorting, so import order
  is enforced rather than a matter of taste — let ruff arrange imports instead of
  hand-placing one and then arguing with the linter about it.
- **`tests/` is excluded from linting on purpose** (`exclude = ["tests"]`). Don't lint,
  reformat, or reorder anything beneath it.
- The coverage gate lives in `[tool.coverage.report] fail_under`. Don't pass
  `--cov-fail-under` on the command line — it silently overrides the file.

---

## First principle: express the intention

Everything below is downstream of one idea. **Code is written once and read many times,
and the reader is usually the person who wrote it, long after the reasons have gone.**
Write so the intention survives: what this is for, why it is this way, what would break
if it changed.

For a project that spends its life pulling signal out of noise, the analogy holds. What
is obvious while writing is a clean signal; memory turns it into noise within months.
What gets written down clearly now is the part that survives the trip.

What that outranks, in practice:

- A name that says what a thing is for beats a shorter one.
- A structure someone can follow beats a cleverer one.
- An explicit note about *why* beats an inference the reader is left to make.
- Plainly readable code beats faster code, up to the point where speed genuinely
  matters — and where it does, the comment should say so.

**Tests express intent too, and are often the better place for it.** A test states what
behaviour is supposed to hold, in a form that fails loudly when someone breaks the
assumption — where a comment making the same claim just goes quietly out of date. When
the thing to convey is an identity, an equivalence, or a boundary condition, a test
usually carries it better than prose: the equivalence tests behind the pulse-train
summation don't merely check the optimization, they record the claim that makes it
valid. Reach for a test whenever the explanation is really a claim about behaviour.

## Don't overdrive your headlights

The same principle, applied at the moment of writing rather than to what's left behind.
Getting the code right is only half the job. **The developer has to be able to tell that
it's right.** A change whose correctness can only be judged by someone with deeper
expertise than the person reviewing it isn't finished, however correct it happens to be.
It has outrun the headlights: a mistake inside it stays invisible until it quietly
produces a wrong measurement months later.

So **don't build something the reader couldn't check afterwards without first explaining
it and confirming that it landed.** If a technique is unfamiliar, explain what it does,
why it applies here, and what would look wrong if it were broken — then implement it.
Explaining after the fact is not the same thing; by then the code exists and the natural
pull is to nod it through.

This is also how to size comments. **Comment density tracks how hard a block is to
verify, not how long or how clever it is.** A loop anyone can read needs nothing. A line
resting on a non-obvious identity — that convolution equals correlation for a palindromic
kernel, say — needs enough that a reader can confirm the reasoning without rederiving it
from scratch.

Practical consequences:

- Prefer an approach the reader can follow over a marginally better one they can't.
- When a clever approach genuinely wins, make it checkable: pin it against a naive
  reference implementation in a test, the way the pulse-train summation and the
  fftconvolve scan are both pinned to direct-summation equivalents. An optimization with
  an equivalence test behind it is verifiable by anyone; the same optimization bare is a
  matter of trust.
- Silent cleverness is a defect even when it computes the right answer.
- If a request would mean going faster than this allows, say so and explain first rather
  than quietly complying.

## Structure

- **Separation of concerns is the highest architectural priority.** Focused modules,
  one responsibility each. This codebase was deliberately dug out of a god object and
  should not drift back.
- **Constructor injection** for collaborators, so everything is testable without patching.
- **Push, don't poll.** Components publish state changes to their listeners rather than
  reaching into another component to read its state — a lock is an event, not a level,
  and a poller misses any event that begins and ends between two polls. Publish from the
  single place the change actually happens (`ContinuousAnalyzer._transition()`), wrap
  each listener in its own `try`/`except` so one failure can't stop the primary job, and
  keep listener bodies trivial: they run on the publisher's thread, so real work like
  disk I/O belongs on the consumer's own thread.
- **Pluggable external services get an ABC plus a concrete implementation** — see
  `WeatherClient` / `CumulusMXWeatherClient`. New backends slot in behind the interface.
- **One owner for state transitions.** In the analyzer state machine, the tier methods
  return the state they think should come next and `_run` performs every transition.
  Don't scatter transitions across the methods that detect the conditions.
- **Readability beats pattern purity.** A null-object class and a plain `if` are both
  acceptable ways to make a feature optional; pick whichever is easier to follow.
- **Optional features are config sections with `enabled = false`,** and a matching
  command-line switch where it makes sense. Config is nested TOML by section
  (station, audio, weather, upload, record, …).
- **CSV is an append-only contract.** New columns go where they won't disturb parsing of
  existing rows, and readers must tolerate their absence in older files.
- **Break up long or convoluted methods** into private helpers named so that the original
  method reads like an English procedure of what it does.
- Keep coverage exclusions surgical — exclude the one unrunnable loop, not the whole file.

## Naming

- Method and variable names must convey intent and usage. Short-lived throwaways
  (loop indices and the like) are excused; nothing else is.
- **Be consistent across the codebase.** Don't say "latest" in one place and "last" in
  another for the same concept — pick one term and use it everywhere.
- Private helper names should read like prose at the call site.
- Anything that can be private should be `_`-prefixed.
- **De-magickify constants.** A bare `3` or `15` in an expression gets a name and a
  comment explaining where the value came from, or gets derived from something already
  in config. If the value was arbitrary, say so in the comment — that's honest and
  useful to the next reader.

## Type hints

Every method gets them. Untyped code is not acceptable here.

- `X | None`, never `Optional[X]`.
- Define a module-level alias when a union repeats — see `CsvColumn` for `float | str`.
- Parametrize container types; avoid bare `dict` and `list`. The exception is when the
  full parametrisation would be so long it obscures more than it explains.
- Avoid `Any` and untyped `obj` wherever something more specific is available.
- Don't over-alias what's already idiomatic. `Path | str` for a pathname is clearer
  as-is than hidden behind a name.

## Optimization

Optimization is genuinely welcome here, and proactive "anything else worth speeding up?"
passes are a standing request. CPU and heat are real constraints — this runs unattended
all day, and a spinning fan counts as a bug report.

Patterns already established, worth reaching for again:

- **Exploit the mathematics before micro-tuning the code.** Correlation against a 0/1
  kernel is just a sum at the 1 positions. Convolution equals correlation when the kernel
  is a palindrome, which is what lets `scipy.signal.fftconvolve` do the scan pass.
- **Hoist invariants.** Anything whose size or value never changes gets computed once in
  the constructor or as a module constant — not per call, and not behind an `lru_cache`
  when a plain attribute will do.
- **Vectorize.** No Python-level loops or list comprehensions over sample arrays.
- **`@njit` numeric helpers** that work only on NumPy values and don't touch Python
  objects — but not when it would cost testability or clarity for a marginal gain.
- **Cheap check first, expensive confirmation second.** Re-acquisition runs a short,
  narrow probe and only escalates to the full FFT search when the probe suggests there's
  something there.
- **Hold resources open** rather than reopening them per use — a persistent audio stream
  over repeated open/close, one SSH connection over two.
- **Random access over draining.** Read what you need out of the ring buffer at the
  offsets you need it; don't consume it.
- **Round, don't truncate,** when staying in phase — and not banker's rounding.
- **Don't refresh faster than useful.** ~20 ms (a Windows scheduler quantum) is the
  practical floor for display updates.

The limits are as firm as the goals. **Fixing something wrong and making something
correct finer are different propositions.** Defects — sign errors, misaligned sampling
grids, biased estimators, quantization that destroys weak-signal resolution — get
enthusiastic approval. Refinements to an already-correct measurement get scepticism.
When proposing one, lead with the practical question whose answer it changes; if there
isn't one, say so and recommend against it. Sub-sample interpolation was declined on
exactly those grounds and should not be re-proposed without a new reason.

Optimization is also where headlights get overdriven most easily, because the win is
real and the reasoning is mathematical. Explain the identity before exploiting it, and
leave an equivalence test behind.

## Tests

- Cover every method of any appreciable complexity. Separate tests into discrete files
  and classes by subject.
- **Write tests to document, not only to catch regressions.** A test name and its
  assertions should say what behaviour is intended and why it holds. Where a knowledge
  gap makes code hard to trust — a mathematical identity, an equivalence between a fast
  path and an obvious one, a boundary that must not move — a test is the durable way to
  state it. See "First principle: express the intention".
- Golden files and generated sample audio in `tests/resources/` pin down DSP behaviour
  so later tweaks have to be deliberate.
- When a refactor makes something testable that wasn't before, write the test then.
- `tests/conftest.py` sets `NUMBA_DISABLE_JIT=1` so `@njit` function bodies are visible
  to coverage. Without it every JIT-compiled function reads as untested no matter how
  well exercised, which is what held the achievable gate down before.
- **Integration tests assert properties, never exact numbers** — "locked within N
  seconds", "position advanced monotonically". They run over real threads and real time,
  so exact assertions will flake on a loaded runner and then get ignored.
- Run them with `pytest -m integration --no-cov`. The `--no-cov` matters: the 97% gate is
  calibrated against the unit suite, and measuring a different subset against it fails
  spuriously.
- Integration tests complement running the program by hand; they don't replace it.
  A lit button or a timer starting at the wrong number needs a person looking.

## Comments and documentation

Match the voice already in the codebase: concise, factual, plain. Avoid AI-assistant tics
— no "X IS REAL", no "it isn't X, it's Y" constructions, no breathless framing.

- **Comment the why, not the what** — especially where a deliberate choice looks wrong
  at a glance. Worth preserving: peak amplitude rather than RMS (impulse noise, not sine
  waves); the noise floor deliberately excluding the impulse so the comparison is
  meaningful; signal and noise-floor phase drifting independently; `gc.disable()` around
  plotting; the `-128` sentinel where `log(0)` would blow up.
- **Anticipate the reader's objection.** If someone would reasonably ask "why didn't you
  just do it the obvious way?", answer that in the comment.
- **Loud comments for genuinely weird necessities** — workarounds for upstream bugs need
  to explain themselves and say what would let them be removed.
- Every module gets a docstring. Every public method over ~10 lines gets one.
- Config options get inline comments explaining what they do, and the sample config is
  documentation in its own right. Keep personal values out of defaults and examples.
- **Don't over-explain.** Cut background the reader didn't ask for; don't justify a
  recommendation in the README that only needs stating. When in doubt, shorter. This
  trims prose that restates the obvious — it does not license leaving hard-to-verify
  code unexplained. See "Don't overdrive your headlights"; the two rules meet at
  whether a reader can confirm the code is correct.
- Where a comment explains the physics or the radio behaviour behind a decision, the
  reasoning is authoritative and the wording is not. Tighten the prose; don't quietly
  change what it claims.
- **Never invent facts** — crash frequencies, dates, history, measurements. If it isn't
  known, say so.
- This project is meant to be educational as well as functional, and gets used as
  reference material. Comments have an audience beyond whoever is editing the file.

## Working style

- For a sizeable new feature, ask clarifying questions before writing any code.
- Review passes are a standing request, and are most useful from an explicit
  perspective — a seasoned DSP engineer, a Python developer with five years'
  experience, a ham with little Python. Present findings as a **numbered list**, each
  with enough detail to be judged on its own, since the response is normally to accept
  some and decline others.
- **Assume a reader fluent in Python who understands DSP fundamentals but is not a DSP
  expert.** Ordinary Python needs no explanation, and neither do the basics — FFTs,
  windowing, dB, Nyquist. The less common constructs warrant a deeper treatment, in the
  comments and in review discussion alike: matched filtering, phase tracking,
  least-squares fitting, the correlation identities this code leans on. Radio knowledge
  is a different matter; see Domain reference.
- Display work is tuned by eye against a real signal, in terms of pixel dimensions,
  padding, and colour ranges. Expect iteration, and change one thing at a time so each
  round of feedback is attributable.

## Domain reference

**Assume an experienced amateur radio operator.** Ham and RF terminology can be used
without gloss, in comments, docs, and discussion alike — QRM, QRN, QSB, S-units, HF,
SSB, AGC, RF and AF gain, preamp, attenuator, noise floor, dBm, propagation, resonance,
antenna tuner, mag loop. Stopping to define these wastes the reader's time and reads as
condescending. The same goes for the interference itself: arcing hardware, gap
discharge, and the 120 pps signature of a 60 Hz distribution grid are the subject matter
here, not exotica.

Note the asymmetry, since it decides where the explanatory effort goes: **radio knowledge
is assumed; DSP fundamentals are assumed too, but deep DSP expertise is not.** Sample
rates, Nyquist, FFTs, bins, dB and windowing need no introduction. What earns a fuller
explanation is the less common machinery and the reasoning specific to this code — why
the scan kernel has to be a palindrome, what the least-squares drift fit is estimating,
why the noise probe sits where it does.

Why any of it matters, for judging whether a change is worth making: a single arcing gap
can lift the noise floor across whole HF bands and make weak-signal work impossible, and
the tool exists to document that well enough for a utility to act on. Burst duration is
a severity measure independent of receiver gain, antenna, and propagation path — unlike
raw amplitude, which is not.

- S-meter: S9 = −73 dBm, 6 dB per S unit. Above S9 the scale changes, so S9+10/+20/+40/+60
  are not linearly spaced.
- Receiver setup for measurement: AGC off, widest filter, SSB, preamp and attenuator off,
  tuned where the antenna is resonant with the tuner bypassed.
- The QRM pulse is a 2.5–6 ms broadband burst, not an impulse — a gap fires continuously
  for as long as instantaneous line voltage exceeds its breakdown threshold, which is
  why the envelope is a symmetric "football" and why the phase lock is so solid.
