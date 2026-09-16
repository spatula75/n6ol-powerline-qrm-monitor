---
name: prose-check
description: Check docstrings, comments and user-facing strings after writing or changing them.  Run this on any file whose prose you touched, before reporting work as finished and before every commit.  Catches the fragments, garden paths and dangling pronouns that ruff and ste_lint cannot see.
---

# Checking prose

Two halves.  The mechanical half is a command and catches what a tool can catch.  The
reading half is short, scoped, and catches what no tool can.

Run both.  A clean `ste_lint` says nothing about fragments, passive voice, or an `-ing`
form used as the main verb.

## The mechanical half

    python tools/ste_lint.py --changed
    python tools/ste_lint.py --fragments <files>

`--changed` must exit clean.  `--fragments` is advisory: it finds participle openers
such as "Measured on this hardware, ..." and is blind to noun-phrase openers.  Read what
it names and expect roughly half to be false alarms.

## Read the first sentence of every paragraph

**This is the whole reading job.** Every fragment found in the review that produced
this skill, seven of seven, was a paragraph-opening sentence.  None was anywhere else.

The reason: at a paragraph opening you are announcing a topic, and a topic announcement
is naturally shaped like a heading, which is a noun phrase.  Then the sentence continues
into a clause and never goes back to give the heading a verb.  Every one of these is a
heading that grew a tail, which is why it reads well while you write it.

    "Two reading modes, and the difference is deliberate rather than historical."
    "A classmethod rather than a static one, because this is an alternative..."
    "Safe to call more than once, and it will be."
    "Snapped here rather than left to the driver, because a V4 cannot..."

Extract them, so this is a list rather than a resolution:

```bash
python - "$@" <<'EOF'
import ast, pathlib, sys
for f in sys.argv[1:]:
    tree = ast.parse(pathlib.Path(f).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            continue
        d = ast.get_docstring(node, clean=True)
        if not d:
            continue
        name = getattr(node, 'name', '<module>')
        for para in [p for p in d.split('\n\n')[1:] if p.strip()]:
            first = ' '.join(para.split())
            print(f'{f}:{name}: {first[:100]}')
EOF
```

That is about 50 sentences for a 700-line module, against 130 body sentences.  Of each
one ask: **does it have a subject and a finite verb?** A docstring's own summary line
is exempt, because a noun-phrase summary is house style here and appears 123 times
against 39 verb-first.

## Then check those same openers for five more things

One line each.  `CLAUDE.md` carries the worked examples.

- **Garden path.** A long gerund subject whose verb reads as a noun. "Viewing a float64
  array as complex128 pairs consecutive values" parses as *complex128 pairs* until it
  does not.
- **Trailing `, which is ...`**, especially twice running, so the point always arrives
  last.
- **A pronoun with a nearer candidate** between it and its antecedent.  Name the noun.
- **A fact with no consequence.** Say what follows from it in the same breath.  This is
  the single change that most improves a hard paragraph.
- **An unqualified domain noun** in a module that does not establish it.  Bare "sweep" is
  fine in `gain_sweep.py` and ambiguous in `sdr_device.py`.

## What this is not

Do not try to automate the reading pass.  Detecting "no finite verb" needs a verb list with no
end.  Measured on this repo: a narrow regex for the exact tell produced 194 hits of
which about three were real, which is worse than `--fragments`.  The reading pass is
cheap only because it is scoped to paragraph openers.  Keep it scoped.
