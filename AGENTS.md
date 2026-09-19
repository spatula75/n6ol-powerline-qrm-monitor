# Project instructions

Before working, read the complete [CLAUDE.md](CLAUDE.md) in this repository.
Its project rules apply to Codex as well as Claude.  Read the files directly:
a reference here does not automatically load their contents.  If a tool truncates
the text, read the remaining sections separately.

Also read [docs/ste-writing.md](docs/ste-writing.md) and
[docs/concepts/agentic-ai.md](docs/concepts/agentic-ai.md).
Use the [prose-check skill](.agents/skills/prose-check/SKILL.md) whenever you change
prose.  Its worked examples remain in CLAUDE.md.

## Working with the author

The author owns the design.  Explain proposed design changes before implementing
them.  Keep changes small enough to understand and review.

The author writes the published documentation.  Identify factual drift and offer
requested corrections without assuming authorship.  Get approval before writing
anything under docs/.

Assume fluency in Python, amateur radio, and DSP fundamentals.  Explain unfamiliar
techniques and the reasoning specific to this project before using them.

Ask before staging, committing, or pushing changes.

Complete the rule review and applicable verification before reporting work as
finished.  A passing test suite does not replace that review or the reading pass
in prose-check.

## Resuming work

Inspect the current branch, working changes, relevant prior discussion, and notebook
before editing.  Preserve unfinished work.  Continue an explicitly resumed branch:
the fresh-branch procedure in CLAUDE.md applies to new work.

Distinguish proposals, approvals, implementation, test results, and hardware
observations.  Do not present an earlier agent's conclusion as a verified fact.
State what the evidence establishes and what remains uncertain.

Keep project evidence and unfinished work in the existing docs-notebook/ files,
following CLAUDE.md's rules for recording them.  Keep lasting project rules in
CLAUDE.md and ask before adding a new rule.  Do not maintain another copy here.
