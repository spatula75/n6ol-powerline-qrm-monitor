# CLAUDE.md

Working notes for Claude Code on this project. Distilled from prior sessions.

## What this is

A continuous powerline QRM monitor for ham radio operators. It samples receiver audio,
locks onto the 120 pps (60 Hz grid; 100 pps where mains is 50 Hz) impulse train from
arcing hardware, and logs signal and noise-floor levels to CSV, charts, and optional
`.wav` event recordings. There is a live Qt display: waterfall, phase-synchronised
oscilloscope, and S-unit bar graphs.

It is a **field tool, not lab test equipment**. The goal is knowing when the utility is
making a racket, not chasing the last decibel. Precision work that costs clarity or
testability is usually not worth it - say so rather than building it.

## Layout

- `lib/buzz/` - the application. One responsibility per module.
- `tools/` - standalone diagnostic scripts.
- `tests/` - unit tests. `tests/integration/` is markered and deselected by default.
- `configure.py`, `level_meter.py` - end-user setup helpers at the root.
- `docs/`, `README.md`, `README-analysis.md` - user- and design-facing docs.
- `templates/index.html` - Jinja2 template for the published web page.
- `config.example.toml` - the documented sample config. Real config lives at
  `~/.buzz/config.toml`; the app falls back to dataclass defaults when it's absent.
- `tmp/` - gitignored scratch space. Notes and bug reports live here, uncommitted.

## Environment

Python 3.12+ (`requires-python = ">=3.12"`). CI pins 3.12; the local dev venv is
`.venv314` on Python 3.14. `lib/` is not installed as a package - it has to be on the
import path. `tests/conftest.py` inserts it for the suite; for a manual run set
`PYTHONPATH=lib` (PyCharm run configurations have `lib` as a source root instead).

Runtime deps are pinned in `requirements.txt`; `pyproject.toml` carries the looser
ranges for packaging. `numpy` is capped below 2.5 by numba's own constraint - check
that ceiling before bumping either.

**Prefer Bourne shell to PowerShell** where a Bourne shell is available, and don't mix
the two within a task - the commands in this file are written for it, and switching
between them mid-task means two sets of quoting rules and two sets of surprises.

PowerShell's surprises are the reason, and they are quiet ones. `Measure-Object -Line`
does not count blank lines, so a line count of a Markdown file comes back plausible
and wrong (385 against a true 467). A native command's stderr is wrapped into
ErrorRecords, so a `git checkout` that succeeded is rendered as a `NativeCommandError`
with `$?` false. Windows PowerShell 5.1 has no `&&` or `||`, so chaining needs
`; if ($?) { }`. None of these announce themselves.

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
`--mute`, `--playback-gain DB|auto`, `--render FILE.mp4`. Playback replays a recorded
`.wav` through the whole pipeline and suppresses CSV, plots, uploads, and recording.
`--render` needs ffmpeg, implies `--playback-gain auto`, and with `--headless` paints
offscreen and implies `--mute`.

**Use `--top` whenever starting a real render during a session.** A render opens a
window and plays through in real time; without always-on-top it can come up behind the
editor, and the operator is then listening to audio from a window they cannot see,
with no way to tell whether it is working. Only skip it when deliberately testing the
headless path.

The coverage gate lives in `pyproject.toml` as the single source of truth - don't pass
`--cov-fail-under` on the command line, it silently overrides it.

## CI and release workflows

Both live in `.github/workflows/` and share the `.github/actions/setup` composite
action, deliberately: a release is verified in the same environment every PR is tested
in, so the two cannot drift apart.

`ci.yml` runs on push and PR to `main`, as two parallel jobs:

- **test** - ruff, then `pytest tests/ --cov`, then the whole suite again with
  `NUMBA_DISABLE_JIT=0`. That second run is not redundant: the interpreted and JIT'd
  paths are not automatically equivalent - NumPy's NEP 50 promotion applies to mixed
  float32/float64 arithmetic in the interpreter while Numba types the expression
  itself, and `average_pulse_amplitude` returned measurably different answers under the
  two before an explicit cast fixed it.
- **integration** - a separate job rather than a later step, so a PR doesn't wait for
  both end to end.

`release.yml` runs on a pushed semver tag (or `workflow_dispatch` with a tag), also as
two jobs:

- **verify** - the gate. Confirms the tag, `lib/buzz/__init__.py`, and `pyproject.toml`
  all state the same version, and that `CHANGELOG.md` has a `## [x.y.z]` section; then
  everything CI runs, plus the integration suite.
- **publish** - `needs: verify`, so no archive exists until the gate passes. Builds
  `.tar.gz` and `.zip` from `git archive` against the tag with a version-stamped prefix,
  writes `SHA256SUMS` beside them, lifts that version's changelog section as the release
  notes, and publishes with `gh release create --verify-tag`.

`git archive` ships exactly what is committed at the tag - no build artefacts, no stray
working-copy files, and `.gitignore`d paths cannot leak in. These archives exist
alongside GitHub's own generated "Source code" ones because GitHub builds those on
demand and has changed their compression before now, silently altering published
checksums; these are built once and stay true.

Headless Linux needs `libportaudio2` for sounddevice and `libegl1 libgl1 libxkbcommon0`
for PySide6 - QtGui links against those at *import* time, so even pure-function tests
that never create a `QApplication` need them.

## Architecture

Threads and a ring buffer, not a call chain. Understand this before changing anything
in the audio or analysis path:

1. **Audio thread** - reads the input device continuously into a ring buffer and
   publishes that data is available. Nothing else touches the device.
2. **Analysis thread** (`ContinuousAnalyzer`) - reads the buffer at the offsets it
   wants, locks onto the pulse train, tracks phase drift by least squares, and
   publishes state transitions and results to registered listeners.
3. **Consumers** - the Qt display (waterfall, scope, bar graphs), the event recorder,
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
- Commit granularity varies - sometimes one commit per unit of work, sometimes a single
  sweep for a batch of PR fixes. Ask which is wanted.
- Note `CHANGELOG.md` under `[Unreleased]` as part of the work, not at release time.
- **No self-attribution in anything that ships to GitHub** - a commit message, a PR
  title or body, a review comment, an issue. No `Co-Authored-By` trailer, no session
  link, no mention of Claude or Claude Code at all, unless there is a real reason to
  name the tooling. Nobody cheers themselves on in their own commits, and Claude Code
  is a tool used to write them, not a byline.

### Commit messages

**A sentence or two per significant change, and nothing at all for the insignificant
ones.** Say what changed and why it was worth doing, then stop. The blow-by-blow
belongs in `CHANGELOG.md`, which is where a reader goes looking for it; repeating it
in the commit buries the one line that mattered.

The cost compounds at the merge, because GitHub's squash stacks every branch commit
message into a single body underneath the PR description. Two commits on `main` run
to 707 and 570 lines against a median of 10. Nobody reads those, so the reasoning in
them was written for nobody.

Where a decision truly needs justifying, the comment beside the code outlives the
commit message and sits where the next reader is already looking. Put it there.

### Releasing

`CONTRIBUTING.md` has the procedure for humans; this is what not to get wrong.

Branch name is `prepare-release-<version>` - `prepare-release-1.4.0`, not
`prepare-release` bare. Get this wrong and either rename the GitHub branch through
its own web UI (Settings > Branches, or the branch list's rename action) so the open
PR migrates with it, or just open a fresh PR against the correctly-named branch -
renaming via `gh api .../branches/{branch}/rename` deleted the old ref and
auto-closed the PR outright instead of migrating it, despite what its docs suggest.

1. **Bump the version in both `lib/buzz/__init__.py` and `pyproject.toml`.** Two files,
   easy to half-do, and the release workflow fails the release if they and the tag
   disagree.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new `## [x.y.z] - date`
   heading. That section is lifted verbatim as the release notes, so write it for
   whoever arrives at the release page.
3. Run `python tools/release_render_check.py` against a recent recording before
   merging. It is not in CI - CI has no live radio, so it cannot produce a recording
   that means anything - which is exactly why skipping it is easy to justify and
   wrong to do: it is the one check that renders a real, current capture rather than
   a synthetic one, at every sample rate the monitor supports.
4. Merge as a PR, then tag the merge commit and push the tag: `git tag 1.3.0 &&
   git push origin 1.3.0`. Plain semver, **no leading `v`** - the workflow's tag filter
   won't match one.

Pushing the tag is what publishes. **The tag goes up before the tests run**, so a failed
verification leaves a tag with nothing attached - which is the intended failure mode, not
a bug. Recover by deleting the tag (`git push --delete origin 1.3.0`), fixing, and
tagging again.

## Before every commit

Judgement calls first, cheapest mechanical check next, most expensive last, human eyes
after all of it. In that order:

1. **Dead code.** While you are already in the area, look for anything certainly
   unused - a function nothing calls, a branch nothing reaches, a path an earlier
   refactor left behind - and ask before removing it, ahead of the checks below. Ask
   rather than delete unasked and ask rather than leave it: whether it is truly dead or
   a hook for something not yet wired up is the user's call, not one to guess at.
2. **Lint** - `ruff check .`. Runs before the test suite, every time, because it is
   fast and cheap where tests are not - there is no reason to wait minutes for a test
   failure that a two-second lint pass would already have caught. Must come back
   completely clean, not merely free of *new* complaints. Fix what it reports rather
   than suppressing it.
3. **Unit suite with coverage** - `pytest --cov`. Must pass, and coverage must stay at
   or above the 97% gate. Running plain `pytest` without `--cov` hides a coverage
   failure that CI then catches; that has broken the build before.
4. **Integration tier**, when the change touches audio, analysis, playback, recording,
   or the display - `pytest -m integration --no-cov`. CI runs it on every PR, so
   catching a failure locally is cheaper than catching it on GitHub.
5. **Hands-on verification.** Changes to the audio path, the display, or the charts get
   checked against a live radio before being committed - green tests are not the finish
   line for those.
6. **Documentation drift.** A code or behaviour change means checking `README.md`,
   `README-analysis.md`, and `config.example.toml` for anything the change makes wrong -
   a described default that moved, a number that no longer holds, a flag or setting that
   changed shape. Docs go stale exactly like comments do, and nothing else catches it;
   there is no test that fails when a README goes out of date.
7. **Diff artifacts.** Read the actual diff before staging, not just the file as it
   ends up. Editing in passes leaves residue that runs and lints clean and is only
   visible in the diff itself: a doubled blank line where a tool split one edit into
   two, a comment whose sentence now trails off because an insertion fell inside it
   rather than before or after, a line break that made sense against the old text and
   reads as a non sequitur against the new. One of this file's own rules briefly read
   "...its own copy - which is how" as an orphaned line-ending, caught only by rereading
   the rendered result, not by any tool. Nothing enforces this but looking.

Then ask before staging anything; see Git workflow.

A locally green run still isn't the whole story: CI also runs the entire suite with
`NUMBA_DISABLE_JIT=0` against the compiled Numba path, which does not always agree with
the interpreted one.

### Lint rules

- **120 columns.** Set as `line-length` under `[tool.ruff]` in `pyproject.toml`. Not 80
  - that limit chops up otherwise readable lines for no benefit on a modern screen.
- Ruff is configured for `E`, `F`, `W`, and `I`. `I` is import sorting, so import order
  is enforced rather than a matter of taste - let ruff arrange imports instead of
  hand-placing one and then arguing with the linter about it.
- **`tests/` is excluded from linting on purpose** (`exclude = ["tests"]`). Don't lint,
  reformat, or reorder anything beneath it.
- The coverage gate lives in `[tool.coverage.report] fail_under`. Don't pass
  `--cov-fail-under` on the command line - it silently overrides the file.

## Reporting finished work

### Summarise structural changes

When a sizeable unit of work is done, **describe what changed structurally** before
anything else: what was added, where it lives, and what it does. New modules, new
classes, new config sections, new flags, new test files - each named, placed, and
explained in a sentence. Say what calls it and what it depends on.

Length is not the constraint here; being skippable is. A reader who has not been
watching every edit needs enough to hold the shape of the change in their head, and to
know where to look when something goes wrong in three weeks. This is the same
principle as "don't overdrive your headlights", applied after the fact: work the
reader cannot navigate has outrun them just as surely as work they cannot verify.

Cover the changes that were *not* obvious from the request as well - a fix made along
the way, a message reworded, a dependency added - and say plainly where something was
left undone or where a test does not guard what it appears to.

### PR descriptions and review comments

**Be brief here - the opposite of the above.** A PR body is read by someone deciding
where to spend their attention, so point at the few things that matter: the centrepiece
of the change, the critical fix, the one decision worth arguing about. Whatever a
reviewer would regret skimming past.

Not a changelog and not a file listing - the diff already says what changed and
`CHANGELOG.md` carries the detail. Same budget as a commit message: a sentence or two
per significant change. The PR says what to *look at* and why it is the important
part.

**Then annotate the diff itself.** Prose in the description is not the same as a note
sitting on the line it is about. After opening a PR, leave **inline review comments**
on the handful of places that matter: the crux of the change, a fix whose reasoning is
not visible from the code, a decision a reviewer might otherwise silently disagree
with, a test that guards something subtle. A few, not a running commentary - every
comment spends attention that the important ones need.

`gh pr review` cannot do this; it only posts a top-level body. Line-anchored comments
go through the API, as one review carrying several:

    gh api --method POST repos/OWNER/REPO/pulls/N/reviews \
      --input review.json

    # review.json
    { "event": "COMMENT",
      "body": "optional overall note",
      "comments": [
        { "path": "lib/buzz/render.py", "line": 128, "side": "RIGHT",
          "body": "why this line is the one to look at" }
      ] }

`line` must be a line the diff actually touches, `side` is `RIGHT` for added lines and
`LEFT` for removed ones, and `start_line` with `line` spans a range. A comment on an
untouched line is rejected, so anchor to the change rather than to its surroundings.

**Each inline comment is a bold one-line summary, then at most two paragraphs.** Lead
with the claim; keep the single most useful piece of evidence, one measured number
rather than the whole investigation, and drop the rest. Past two paragraphs a comment
stops being read, and the long ones dilute the short ones that matter. The detail almost
always already sits in the code comment beside that very line, which is where a reader
who wants it is looking anyway; the review comment only has to get them to look. Same
reasoning as the error-message rule below: three things, not three paragraphs.

When finishing a piece of work, if you did something **noteworthy** (a non-obvious
architectural pattern, a convention the rest of the code leans on, a footgun future contributors will
hit) **or** established a **general principle** (a "we should always do X" rule that
applies beyond the immediate task), ask whether the lesson should be committed to this
file. If yes, persist it here - prefer extending an existing section over creating a
new top-level one, and match the existing tone: terse, concrete, with code precedents
where useful.

The bar is "noteworthy or principle-establishing", not "everything we did". Routine
bug fixes, narrow refactors, and pure implementation details don't trigger this.
Architectural conventions, recurring patterns, footguns, performance considerations,
and cross-cutting style decisions do. Ask once per noteworthy item, in the wrap-up
moment of that piece of work - not retroactively at the end of a long session for
everything.

**The same moment, look for structure that has started to emerge.** A shape that has
now appeared three times is usually trying to become something, and naming it makes the
next instance obvious instead of improvised. Say what the pattern is and what it buys
before moving any code.

The precedent is `SpectrumGeometry`, `SweepGeometry` and `Loudness`, arrived at
independently: each a frozen dataclass built by a pure factory from config, holding no
Qt and doing no I/O, so the arithmetic is exhaustively testable without a display. That
is a pattern this codebase found rather than imported. Underneath it sits a domain rule
discovered three separate times: **fix the quantity in time, derive the sample count
from the rate**. The FFT window, the ring buffer and the scope phosphor each had it
wrong the same way.

**The counterweight matters as much as the rule.** A small gain does not justify churn,
and *readability beats pattern purity* still stands. Refactor where the structure
actually clarifies, not to make the code resemble a catalogue.

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
- Plainly readable code beats faster code, up to the point where speed truly
  matters - and where it does, the comment should say so.

**Tests express intent too, and are often the better place for it.** A test states what
behaviour is supposed to hold, in a form that fails loudly when someone breaks the
assumption - where a comment making the same claim just goes quietly out of date. When
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
it and confirming that it was understood.** If a technique is unfamiliar, explain what it does,
why it applies here, and what would look wrong if it were broken - then implement it.
Explaining after the fact is not the same thing; by then the code exists and the natural
pull is to nod it through.

This is also how to size comments. **Comment density tracks how hard a block is to
verify, not how long or how clever it is.** A loop anyone can read needs nothing. A line
resting on a non-obvious identity - that convolution equals correlation for a palindromic
kernel, say - needs enough that a reader can confirm the reasoning without rederiving it
from scratch.

Practical consequences:

- Prefer an approach the reader can follow over a marginally better one they can't.
- When a clever approach truly wins, make it checkable: pin it against a naive
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
  reaching into another component to read its state - a lock is an event, not a level,
  and a poller misses any event that begins and ends between two polls. Publish from the
  single place the change actually happens (`ContinuousAnalyzer._transition()`), wrap
  each listener in its own `try`/`except` so one failure can't stop the primary job, and
  keep listener bodies trivial: they run on the publisher's thread, so real work like
  disk I/O belongs on the consumer's own thread.
- **Pluggable external services get an ABC plus a concrete implementation** - see
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
- Keep coverage exclusions surgical - exclude the one unrunnable loop, not the whole file.

## Naming

- Method and variable names must convey intent and usage. Short-lived throwaways
  (loop indices and the like) are excused; nothing else is.
- **Be consistent across the codebase.** Don't say "latest" in one place and "last" in
  another for the same concept - pick one term and use it everywhere.
- Private helper names should read like prose at the call site.
- Anything that can be private should be `_`-prefixed.
- **De-magickify constants.** A bare `3` or `15` in an expression gets a name and a
  comment explaining where the value came from, or gets derived from something already
  in config. If the value was arbitrary, say so in the comment - that's honest and
  useful to the next reader.
- **A constant used by two or more modules belongs in `buzz/constants.py`, not
  redefined in each.** `FULL_SCALE_COUNTS`, `S9_DBM`, and `DB_PER_S_UNIT` live there
  because dsp.py, scope.py, device_setup.py, waterfall.py, plotter.py, and
  level_meter.py all need the same numbers and previously each defined its own copy,
  which is how device_setup.py's level bar drifted to 4.75 dB/segment while every
  S-meter in the program stayed at 6, with nothing to notice the two had come apart.
  When auditing for duplicates, check root-level scripts (`level_meter.py`,
  `configure.py`) as well as `lib/buzz/` - a first pass that grepped only the package
  missed `level_meter.py`'s own copy of `S9_DBM` entirely. Move a constant here the
  moment a second module needs it, not before: a value only one module uses belongs
  with that module, where a reader finds it without a second file to check.
  Derive dependents from the shared constant (`_BAR_WIDTH = round(20 * log10(...) /
  DB_PER_S_UNIT)`) rather than hand-computing a literal that can drift again the same
  way, and pin the relationship with a test - see `tests/test_constants.py`.
  `buzz.dsp.SILENCE_DBFS` and `PULSE_WIDTH_SAMPLES` are already this kind of shared
  constant and stay in `dsp.py` rather than moving: dsp is imported everywhere that
  needs them, so a re-export would only add a second name for the same thing.

## Type hints

Every method gets them. Untyped code is not acceptable here.

- `X | None`, never `Optional[X]`.
- Define a module-level alias when a union repeats - see `CsvColumn` for `float | str`.
- Parametrize container types; avoid bare `dict` and `list`. The exception is when the
  full parametrisation would be so long it obscures more than it explains.
- Avoid `Any` and untyped `obj` wherever something more specific is available.
- Don't over-alias what's already idiomatic. `Path | str` for a pathname is clearer
  as-is than hidden behind a name.

## Optimization

Optimization is welcome here, and proactive "anything else worth speeding up?"
passes are a standing request. CPU and heat are hard constraints - this runs unattended
all day, and a spinning fan counts as a bug report.

Patterns already established, worth reaching for again:

- **Exploit the mathematics before micro-tuning the code.** Correlation against a 0/1
  kernel is just a sum at the 1 positions. Convolution equals correlation when the kernel
  is a palindrome, which is what lets `scipy.signal.fftconvolve` do the scan pass.
- **Hoist invariants.** Anything whose size or value never changes gets computed once in
  the constructor or as a module constant - not per call, and not behind an `lru_cache`
  when a plain attribute will do.
- **Vectorize.** No Python-level loops or list comprehensions over sample arrays.
- **`@njit` numeric helpers** that work only on NumPy values and don't touch Python
  objects - but not when it would cost testability or clarity for a marginal gain.
  **Always with an explicit signature** - see below.
- **Cheap check first, expensive confirmation second.** Re-acquisition runs a short,
  narrow probe and only escalates to the full FFT search when the probe suggests there's
  something there.
- **Hold resources open** rather than reopening them per use - a persistent audio stream
  over repeated open/close, one SSH connection over two.
- **Random access over draining.** Read what you need out of the ring buffer at the
  offsets you need it; don't consume it.
- **Round, don't truncate,** when staying in phase - and not banker's rounding.
- **Don't refresh faster than useful.** ~20 ms (a Windows scheduler quantum) is the
  practical floor for display updates.

The limits are as firm as the goals. **Fixing something wrong and making something
correct finer are different propositions.** Defects - sign errors, misaligned sampling
grids, biased estimators, quantization that destroys weak-signal resolution - get
enthusiastic approval. Refinements to an already-correct measurement get scepticism.
When proposing one, lead with the practical question whose answer it changes; if there
isn't one, say so and recommend against it. Sub-sample interpolation was declined on
exactly those grounds and should not be re-proposed without a new reason.

Optimization is also where headlights get overdriven most easily, because the win is
real and the reasoning is mathematical. Explain the identity before exploiting it, and
leave an equivalence test behind.

### Numba: always compile eagerly

**Every `@njit` gets an explicit signature**, so numba compiles at import instead of on
first call. Take the time to work out the right types; it is not optional and it is not
a micro-optimisation.

Lazy compilation runs on whichever thread reaches the function first. In this program
that is the Qt thread, part-way through a paint, while audio keeps arriving on another
one - so the window locks for about a second at the worst possible moment. It put a
frozen second at the head of every rendered video and made the display ignore the
opening of every replay. Declaring the signatures took that from 1.1 s to 0.23 s.

Pair it with `cache=True` so the compiled code survives between runs and only the first
run after a source change pays at all.

**Work the types out from the running program, not by reading.** Reading the code gave
the wrong answer twice: the array looked like `float64` or `int16` and is in fact
`float32` every time - `np.abs(int16 samples - a float32 DC estimate)`. The reliable
method:

1. Leave the decorator lazy for a moment and exercise the real paths.
2. Read `fn.signatures` off the dispatcher - numba lists exactly what it compiled.
3. **Check it against production, not the test suite.** The suite compiled three
   signatures; two were test artefacts. Wrap the function and log dtypes while the real
   analyser runs.
4. **Cover every source.** A replayed `.wav` and the live sound card can differ in
   principle, so check both before declaring. Here they agreed - that is a result, not
   an assumption.

A declared signature also means a caller passing something unexpected fails loudly at
that call rather than quietly compiling another variant mid-flight.

## Tests

- Cover every method of any appreciable complexity. Separate tests into discrete files
  and classes by subject.
- **Write tests to document, not only to catch regressions.** A test name and its
  assertions should say what behaviour is intended and why it holds. Where a knowledge
  gap makes code hard to trust - a mathematical identity, an equivalence between a fast
  path and an obvious one, a boundary that must not move - a test is the durable way to
  state it. See "First principle: express the intention".
- Golden files and generated sample audio in `tests/resources/` pin down DSP behaviour
  so later tweaks have to be deliberate.
- **When a test stands in for a sound source, it uses the dtype the real source
  produces.** Synthetic audio is easy to build in whatever type is convenient, and the
  convenient one is usually wrong: tests here fed `int32` and `uint32` arrays into a
  path that only ever sees `float32` in production. Nothing failed, because the code
  under test is generic - so the tests quietly exercised type combinations that cannot
  occur, and the one place it mattered (a numba signature) only surfaced it by
  accident. Check what the live pipeline actually produces, and match it. This applies
  to the golden generator too, or the pinned values describe a path nobody runs.
- When a refactor makes something testable that wasn't before, write the test then.
- `tests/conftest.py` sets `NUMBA_DISABLE_JIT=1` so `@njit` function bodies are visible
  to coverage. Without it every JIT-compiled function reads as untested no matter how
  well exercised, which is what held the achievable gate down before.
- **Integration tests assert properties, never exact numbers** - "locked within N
  seconds", "position advanced monotonically". They run over real threads and real time,
  so exact assertions will flake on a loaded runner and then get ignored.
- Run them with `pytest -m integration --no-cov`. The `--no-cov` matters: the 97% gate is
  calibrated against the unit suite, and measuring a different subset against it fails
  spuriously.
- **An integration test isn't finished until you have seen it fail.** Delete the thing
  it is meant to catch and check that it goes red, with the message it was written to
  print. The restart test only earned its keep once removing `analyzer.reset()` from
  the restart path failed it. Tests over real threads pass for the wrong reason more
  easily than most - a wait that returns early and an assertion that never runs both
  look identical to a pass.
  The two rules below are the same failure in tests that already exist and look fine:
  green, well named, guarding nothing. Both were found in review, not by anything going
  red, which is the point.
- **At least one number in an assertion must come from outside the thing under test.**
  `test_the_video_lasts_as_long_as_the_replay_did` was written for the opening-frame bug
  and could not fail: it read the source duration and the video duration from the same
  `.mp4`, where the container's duration *is* the video stream's, so it asserted
  `0.0 < 0.2`. `-shortest` trims the audio to whatever the video came to, so a render
  missing its first second is a perfectly self-consistent file and every internal
  cross-check agrees with itself. It measures the source `.wav` now. Ask which of the two
  figures the buggy code could not have influenced; if the answer is neither, the test is
  decorative.
- **A test must be able to fail.** No test exists to raise the coverage number; every
  one exists to catch a specific way the code could be wrong, and if nothing the test
  does could turn red for a real bug, it is not testing anything. The two failure
  shapes to watch for: a test with no assertion, or one that runs code purely for the
  line-coverage credit, and a test whose assertion is a tautology - checking that a
  mock returned the value it was told to return, rather than that the code under test
  computed something correctly from it. Mocking a boundary (ffmpeg, a device, the
  filesystem) is correct and necessary; the mistake is mocking the computation itself
  and then asserting on the mock's output, which proves only that Python can return a
  value. Before trusting a test, ask what change to the real code would make it fail -
  if the honest answer is "none," delete or rewrite it.
- **Deleting a safeguard means guarding the reason it became unnecessary.** The frame
  padding in `ffmpeg_command()` was removed because the time-pinned FFT window makes the
  bin count 128 at every rate. True, but only because `validate_sample_rate` refuses
  everything below 8 kHz, a coupling between two modules that nothing stated. The seven
  rates already parametrized all sit well inside the band, so all seven would keep
  passing if the band moved; at a 7500 Hz floor, 252 rates become legal at an odd bin
  count and every render at those dies in x264. A removal transfers its job to an
  invariant elsewhere. Write the test that couples the two, sweep the domain
  exhaustively where it is cheap (40,001 rates cost about a second), and name every way
  it can break in the message, since the fix differs and the assertion cannot tell which
  happened. See `test_every_admitted_rate_gives_an_encodable_frame`.
- **Assert on transitions, not on polled state.** Register a listener
  (`ContinuousAnalyzer.add_state_listener`) and assert on the sequence it records; see
  `StateLog` in `tests/integration/harness.py`. Polling `analyzer.state` cannot see a
  lock that dropped and came back between two polls, and that is exactly what restart
  has to prove. Push, don't poll applies to tests too.
- **Integration tests run muted and headless.** Anything that launches the monitor
  passes `--mute` and `--headless` (or sets `QT_QPA_PLATFORM=offscreen`) unless the
  thing under test is the audio device or the window itself. A suite that seizes the
  speakers or throws a window in front of whoever is running it will not be run, and
  neither adds anything to what is usually being tested. Note that `--headless` alone
  does *not* imply `--mute` - only `--render --headless` does - and that headless
  playback never exits on its own, so a test driving it must stop the process rather
  than wait for it.
- Integration tests complement running the program by hand; they don't replace it.
  A lit button or a timer starting at the wrong number needs a person looking.

### Optional features must not hold coverage hostage

An optional dependency is one a user can decline - ffmpeg for `--render` is the
example. **Its absence must not move the coverage number**, or the gate starts failing
on machines that simply chose not to install something, and the fix people reach for
is lowering the gate.

The shape that works, and what `render.py` does:

- **Unit-test against the boundary, mocked.** `shutil.which` and `subprocess.Popen`
  are patched, so every line of the module is exercised with no binary present.
  Measured: `render.py` sits at 96% and `fonts.py` at 100% with ffmpeg entirely absent
  from PATH.
- **Put the real thing in the integration tier**, which runs `--no-cov` and is
  deselected from the unit run, so it cannot affect the number either way.
- **Skip, don't fail, when it is missing** - a contributor who never renders still
  gets a green suite.
- **But make CI fail rather than skip.** A skip is invisible in a green run, so an
  environment that was meant to have the dependency and lost it would quietly stop
  testing that path. `BUZZ_REQUIRE_FFMPEG=1`, set only by the workflows, turns the
  skip into a failure that says which tool is missing and where it was installed from.

If a feature truly cannot be covered without the dependency, find a workaround -
extract the logic behind an interface and test that - rather than letting CI's result
depend on what happens to be installed. A coverage gate that means different things on
different machines means nothing.

## Comments and documentation

Match the voice already in the codebase: concise, factual, plain. Avoid AI-assistant tics:
no "X IS REAL", no "it isn't X, it's Y" constructions, no breathless framing.

**Banned words and punctuation.** These are assistant tells rather than house voice, and
they are banned outright in files and comments:

- **No em dashes.** A single spaced hyphen carries most of what one was doing, and
  ordinary prose should take that: `a field tool - not lab gear`. Reach for `--` only
  where the break really needs the weight, meaning a refinement that also wants a pause;
  the codebase already uses `--` that way in comments. A colon, semicolon, comma or full
  stop is often better than either, and a *paired* aside is nearly always clearer with
  commas, since half of a `- ... -` pair at the start of a line reads as a bullet.
- **These words:** *genuine*, *genuinely*, *load-bearing*, *is real*, *are real*, *land*,
  *lands*, *landed*.

Most of them are doing emphasis rather than work. "A genuine bug" is a bug; "the
load-bearing line" is the line that matters; "the value lands at 128" is the value being
128. Say the thing.

- **Comment the why, not the what** - especially where a deliberate choice looks wrong
  at a glance. Worth preserving: peak amplitude rather than RMS (impulse noise, not sine
  waves); the noise floor deliberately excluding the impulse so the comparison is
  meaningful; signal and noise-floor phase drifting independently; `gc.disable()` around
  plotting; the `-128` sentinel where `log(0)` would blow up.
- **A factual claim in a docstring or comment gets checked before it is committed, not
  taken on faith because it reads plausibly.** A number, a worked example, a count of
  something, a "this is what X computes" - anything a reader could in principle verify -
  actually gets verified: run the formula, read the code it describes, compute the
  example by hand or in a scratch script. An audit of this codebase found a floating-point
  example that was simply wrong (`0.3 * 30` is exactly `9.0` in Python, not the
  `8.999999999999998` the comment claimed), a dB-per-segment figure calculated for the
  wrong constant, and two docstrings still describing a design a refactor had already
  replaced - `scope.py`'s "sweep width equals the phase period" claim, true only at the
  16 kHz default, false in general since the sample-rate-independence work decoupled the
  two. All four read as perfectly reasonable until checked.
- **A change that touches the reasoning a nearby comment depends on means checking that
  comment, not just the code.** A constant that moved, a formula that changed, a design
  that was replaced - each can leave a comment three lines away (or in a different
  module entirely, as with `scope.py`'s stale claim above) stating something that used
  to be true. Re-derive the comment's claim against the new code before moving on, the
  same way a changed API means checking every caller.
- **Anticipate the reader's objection.** If someone would reasonably ask "why didn't you
  just do it the obvious way?", answer that in the comment.
- **Loud comments for truly weird necessities** - workarounds for upstream bugs need
  to explain themselves and say what would let them be removed.
- Every module gets a docstring. Every public method over ~10 lines gets one.
- Config options get inline comments explaining what they do, and the sample config is
  documentation in its own right. Keep personal values out of defaults and examples.
- **Don't over-explain.** Cut background the reader didn't ask for; don't justify a
  recommendation in the README that only needs stating. When in doubt, shorter. This
  trims prose that restates the obvious - it does not license leaving hard-to-verify
  code unexplained. See "Don't overdrive your headlights"; the two rules meet at
  whether a reader can confirm the code is correct.
- Where a comment explains the physics or the radio behaviour behind a decision, the
  reasoning is authoritative and the wording is not. Tighten the prose; don't quietly
  change what it claims.
- **Never invent facts** - crash frequencies, dates, history, measurements. If it isn't
  known, say so.
- This project is meant to be educational as well as functional, and gets used as
  reference material. Comments have an audience beyond whoever is editing the file.

### Error messages

An error message is read by someone who is stuck, usually in a hurry, and often
without the source in front of them. **Every message - raised, logged, or asserted -
carries three things:**

1. **Context.** What was being attempted when it failed, with the specifics filled in:
   the path, the setting, the device, the value. "Not found" is useless; "not found on
   PATH, and --render needs it" is not.
2. **What the error means and what likely caused it,** in plain language. Name the
   probable cause where there is one, and say which of several it might be rather than
   leaving the reader to guess. Avoid restating the exception class in prose.
3. **What to do about it.** A concrete next step - the command to run, the setting to
   change, the file to look in. If the failure is survivable, say what the program did
   instead, so nobody hunts a problem that has already been worked around.

**Three things, not three paragraphs.** Two or three sentences carries all of it; past
that the message stops being read, which costs more than leaving something out. Cut the
reasoning and keep the facts - *why* it works this way belongs in a comment beside the
code, where the person who needs it is already looking.

**Don't end a sentence with a preposition** in anything the user sees - messages, log
lines, `--help` text, README prose. "the figure at which it was recorded", not "the
figure it was recorded at". A made-up rule, and this project follows it anyway; the
house style is formal and the messages should match it. Comments and docstrings are
not user-facing and need not bother.

Precedents worth copying: `find_ffmpeg()` in `render.py` names the missing program, the
one flag that needs it, the winget command that installs it, the config setting that
points at it instead, and that nothing else in the monitor cares. The font test in
`test_fonts.py` names the directory it searched, why the file is loaded from there at
all, what probably moved, and what the display degrades to meanwhile.

This applies to test failure messages as much as to runtime errors. A test that fails
with a bare `assert files` tells whoever hits it that something is wrong and nothing
else; the person reading it is usually not the person who wrote it, and may be reading
it in CI output months later.

## Working style

- For a sizeable new feature, ask clarifying questions before writing any code.
- Review passes are a standing request, and are most useful from an explicit
  perspective - a seasoned DSP engineer, a Python developer with five years'
  experience, a ham with little Python. Present findings as a **numbered list**, each
  with enough detail to be judged on its own, since the response is normally to accept
  some and decline others.
- **Assume a reader fluent in Python who understands DSP fundamentals but is not a DSP
  expert.** Ordinary Python needs no explanation, and neither do the basics - FFTs,
  windowing, dB, Nyquist. The less common constructs warrant a deeper treatment, in the
  comments and in review discussion alike: matched filtering, phase tracking,
  least-squares fitting, the correlation identities this code leans on. Radio knowledge
  is a different matter; see Domain reference.
- Display work is tuned by eye against a real signal, in terms of pixel dimensions,
  padding, and colour ranges. Expect iteration, and change one thing at a time so each
  round of feedback is attributable.

## Domain reference

**Assume an experienced amateur radio operator.** Ham and RF terminology can be used
without gloss, in comments, docs, and discussion alike - QRM, QRN, QSB, S-units, HF,
SSB, AGC, RF and AF gain, preamp, attenuator, noise floor, dBm, propagation, resonance,
antenna tuner, mag loop. Stopping to define these wastes the reader's time and reads as
condescending. The same goes for the interference itself: arcing hardware, gap
discharge, and the 120 pps signature of a 60 Hz distribution grid are the subject matter
here, not exotica.

Note the asymmetry, since it decides where the explanatory effort goes: **radio knowledge
is assumed; DSP fundamentals are assumed too, but deep DSP expertise is not.** Sample
rates, Nyquist, FFTs, bins, dB and windowing need no introduction. What earns a fuller
explanation is the less common machinery and the reasoning specific to this code - why
the scan kernel has to be a palindrome, what the least-squares drift fit is estimating,
why the noise probe sits where it does.

Why any of it matters, for judging whether a change is worth making: a single arcing gap
can lift the noise floor across whole HF bands and make weak-signal work impossible, and
the tool exists to document that well enough for a utility to act on. Burst duration is
a severity measure independent of receiver gain, antenna, and propagation path - unlike
raw amplitude, which is not.

- S-meter: S9 = −73 dBm, 6 dB per S unit. Above S9 the scale changes, so S9+10/+20/+40/+60
  are not linearly spaced.
- Receiver setup for measurement: AGC off, widest filter, SSB, preamp and attenuator off,
  tuned where the antenna is resonant with the tuner bypassed.
- The QRM pulse is a 2.5–6 ms broadband burst, not an impulse - a gap fires continuously
  for as long as instantaneous line voltage exceeds its breakdown threshold, which is
  why the envelope is a symmetric "football" and why the phase lock is so solid.
