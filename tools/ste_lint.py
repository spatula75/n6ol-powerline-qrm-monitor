"""
Check prose against the writing rules in CLAUDE.md and docs/ste-writing.md.

Run this before a commit, over whatever the change touched:

    python tools/ste_lint.py lib/buzz/scope.py tests/test_scope_math.py
    python tools/ste_lint.py --changed              # every line the diff added
    python tools/ste_lint.py --changed --base main

It exits 1 when it finds something, so it can sit beside `ruff check .` in the
pre-commit sequence.

-------------------------------------------------------------------------------
What counts as prose
-------------------------------------------------------------------------------

Rules about sentences have no business being applied to code, so this extracts
prose rather than scanning raw lines:

  * Python comments, through `tokenize`.
  * Python docstrings, through `ast`.
  * The message arguments of `raise`, `assert`, and `logger.*`, which is the
    text a reader meets while something is wrong.  An f-string arrives as
    separate pieces, so those are rejoined with `{}` standing in for each
    value.  Without that, "could not start ffmpeg at {path}" reads as a
    sentence ending in a preposition, and the tool reports a fault it invented.
  * Markdown outside fenced code blocks.

Identifiers, expressions and command syntax are never examined, per
docs/ste-writing.md's own "What this applies to".

-------------------------------------------------------------------------------
Two modes
-------------------------------------------------------------------------------

STRICT covers text a reader meets while stuck: error messages, log lines, and
assertion messages.  Every rule applies, including the length caps, the ban on
semicolons and contractions, and the ban on ending a sentence with a
preposition.

FLAVORED covers comments, docstrings and Markdown.  The sentence discipline
applies; the length caps and vocabulary limits do not.  See ste-writing.md's
"Two modes" for why the two differ.

-------------------------------------------------------------------------------
Where the spacing rule applies
-------------------------------------------------------------------------------

Two spaces after a period is a house rule for prose somebody reads as prose:
comments, docstrings, and end-user documentation.  It is not enforced in
CHANGELOG.md, CLAUDE.md, or schema.json, which are records and machine-readable
sources rather than prose, and which have never followed it - 329 instances at
the time of writing.  Enforcing it there would mean a large cosmetic diff that
improves nothing.  Every other rule still applies to those files.

-------------------------------------------------------------------------------
What this cannot do
-------------------------------------------------------------------------------

It checks the rules that can be checked mechanically.  Three of the rules in
ste-writing.md's self-lint cannot be, and still need a person: sentence
fragments, passive voice where the actor is known, and an `-ing` form used as
the main verb.  A clean run means no mechanical fault was found.  It does not
mean the prose is good, and it cannot make a hollow paragraph true.
"""

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, replace
from pathlib import Path

# Assistant tells, banned outright by CLAUDE.md.
BANNED = re.compile(r'\b(genuine|genuinely|load-bearing|is real|are real|lands?|landed)\b', re.I)
MARKETING = re.compile(
    r'\b(seamless|robust|powerful|cutting-edge|effortless|world-class'
    r'|next-generation|revolutionary)\b', re.I)
# British spellings this project is converting as it touches them; see
# ste-writing.md's override 3.
BRITISH = re.compile(
    r'\b\w*(centre|colour|behaviour|synchronis|quantis|neighbour|analyse'
    r'|organis|licence|initialis|normalis)\w*\b', re.I)
# Prefer the short common word.
WORDY = {
    'commence': 'start', 'initiate': 'start', 'utilize': 'use', 'leverage': 'use',
    'facilitate': 'help', 'ensure': 'make sure', 'ensures': 'make sure',
    'prior to': 'before', 'subsequent to': 'after', 'regarding': 'about',
    'concerning': 'about', 'obtain': 'get', 'acquire': 'get',
    'demonstrate': 'show', 'demonstrates': 'show', 'additionally': 'also',
    'furthermore': 'also', 'moreover': 'also',
}
WORDY_RE = re.compile(r'\b(' + '|'.join(sorted(WORDY, key=len, reverse=True)) + r')\b', re.I)
CONTRACTION = re.compile(r"\b\w+(?:n't|'re|'ll|'ve|'d|'m)\b")
PREPOSITIONS = frozenset(
    ('at', 'by', 'for', 'from', 'in', 'of', 'on', 'to', 'with', 'into', 'about', 'over'))

# ste-writing.md's strict caps: 20 words for an instruction, 25 for a description.
# A message is a description unless it is written as a command, which is a
# distinction this cannot draw, so it uses the looser of the two.
STRICT_WORD_CAP = 25

# Files exempt from the two-space rule only.  See the module docstring.
SPACING_EXEMPT = ('CHANGELOG.md', 'CLAUDE.md', '.json')
# Files that quote the banned words in order to ban them.
RULE_BOOKS = ('CLAUDE.md', 'ste-writing.md')


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    detail: str
    text: str
    strict: bool
    # Last line of the prose this came from.  A docstring spans many lines, and
    # --changed has to notice an edit anywhere inside one, not only on the line the
    # docstring opens.  Reporting by the opening line alone hid a real fault during
    # this tool's own first run against the branch that introduced it.
    end_line: int = 0

    def touches(self, added: set[int]) -> bool:
        """Whether any line of the prose behind this finding was added."""
        return any(line in added for line in range(self.line, max(self.end_line, self.line) + 1))

    def render(self) -> str:
        mode = 'strict' if self.strict else 'flavored'
        excerpt = ' '.join(self.text.split())[:100]
        return f'{self.path}:{self.line}: {self.rule} ({mode}) - {self.detail}\n    {excerpt}'


def sentences(text: str) -> list[str]:
    """Split prose into sentences, leaving decimals and initials intact.

    The negative lookbehind is what keeps "2.5 ms" and "config.py" from being read
    as sentence ends; both appear constantly in this codebase's prose.
    """
    flat = ' '.join(text.split())
    return [part.strip() for part in re.split(r'(?<![A-Z0-9])[.!?]\s+', flat) if part.strip()]


def _check_spacing(path: str, line: int, text: str, strict: bool) -> list[Finding]:
    """One space after a period, where two belong."""
    if path.endswith(SPACING_EXEMPT):
        return []
    for match in re.finditer(r'(?<![A-Z0-9])\.(?= [A-Z])', text):
        before = text[:match.start()].rstrip()
        if before.endswith(('e.g', 'i.e', '.')):
            continue
        # Exactly one space is the fault.  A newline is a paragraph break, and a
        # period inside code such as `self.x` is excluded by the lookbehind above.
        if text[match.start() + 2:match.start() + 3] != ' ':
            excerpt = text[max(0, match.start() - 30):match.start() + 30]
            return [Finding(path, line, 'one space after a period', excerpt.strip(),
                            text, strict)]
    return []


def _check_words(path: str, line: int, text: str, strict: bool) -> list[Finding]:
    """Banned words, marketing adjectives, British spellings, and wordy choices."""
    found = []
    if not path.endswith(RULE_BOOKS):
        for pattern, rule in ((BANNED, 'banned word'), (MARKETING, 'marketing adjective')):
            hit = pattern.search(text)
            if hit:
                found.append(Finding(path, line, rule, f'"{hit.group(0)}"', text, strict))
    hit = BRITISH.search(text)
    if hit:
        found.append(Finding(path, line, 'British spelling', f'"{hit.group(0)}"', text, strict))
    hit = WORDY_RE.search(text)
    if hit:
        better = WORDY[hit.group(0).lower()]
        found.append(Finding(path, line, 'wordy',
                             f'"{hit.group(0)}" -> "{better}"', text, strict))
    return found


def _check_dashes(path: str, line: int, text: str, strict: bool) -> list[Finding]:
    """Em dashes anywhere, and en dashes used as punctuation rather than as a range.

    A tight en dash between two figures is a numeric range, which is typography
    rather than the dash CLAUDE.md bans.  A spaced one is being used as a dash.
    """
    if '—' in text:
        return [Finding(path, line, 'em dash', 'use a spaced hyphen', text, strict)]
    if re.search(r'(?<![0-9A-Za-z])–|–(?![0-9A-Za-z])', text):
        return [Finding(path, line, 'en dash', 'use a spaced hyphen', text, strict)]
    return []


def _check_strict(path: str, line: int, text: str) -> list[Finding]:
    """The rules that apply only to text a reader meets while stuck."""
    found = []
    hit = CONTRACTION.search(text)
    if hit:
        found.append(Finding(path, line, 'contraction', f'"{hit.group(0)}"', text, True))
    if ';' in text:
        found.append(Finding(path, line, 'semicolon', 'write two sentences', text, True))
    for sentence in sentences(text):
        words = sentence.split()
        if len(words) > STRICT_WORD_CAP:
            found.append(Finding(path, line, 'over the strict cap',
                                 f'{len(words)} words, cap is {STRICT_WORD_CAP}',
                                 sentence, True))
        # A sentence closing on an interpolated value ends with the value, not with
        # the word before it.  "Could not start ffmpeg at {path}" is correct prose,
        # and flagging it would push the writer into rewording a good message.
        if re.search(r'\{\}[^A-Za-z]*$', sentence):
            continue
        tail = re.sub(r'[^A-Za-z]+$', '', sentence).split()
        if tail and tail[-1].lower() in PREPOSITIONS:
            found.append(Finding(path, line, 'ends with a preposition',
                                 f'"{tail[-1]}"', sentence, True))
    return found


def check(path: str, line: int, text: str, strict: bool) -> list[Finding]:
    """Every rule that applies to one piece of prose."""
    found = _check_dashes(path, line, text, strict)
    found += _check_words(path, line, text, strict)
    found += _check_spacing(path, line, text, strict)
    if strict:
        found += _check_strict(path, line, text)
    return found


# ---------------------------------------------------------------------------
# Extracting prose
# ---------------------------------------------------------------------------

# A comment that is a directive to a tool rather than a sentence to a reader.
DIRECTIVES = ('noqa', 'type:', 'pragma', '!', 'ruff:', 'fmt:')
LOG_METHODS = frozenset(('debug', 'info', 'warning', 'error', 'exception', 'critical'))
# Below this a "message" is a label or a fragment rather than prose, and the
# sentence rules say nothing useful about it.
MIN_MESSAGE_WORDS = 4


def _strict_strings(node: ast.AST) -> list[tuple[int, int, str]]:
    """The message of one raise, assert, or logger call, rebuilt from its pieces."""
    parts: list[str] = []
    first: int | None = None
    last = 0
    for inner in ast.walk(node):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
            parts.append(inner.value)
            first = first if first is not None else inner.lineno
            last = max(last, inner.end_lineno or inner.lineno)
        elif isinstance(inner, ast.FormattedValue):
            parts.append('{}')
    joined = ''.join(parts)
    if first is None or len(joined.split()) < MIN_MESSAGE_WORDS:
        return []
    return [(first, last, joined)]


def python_prose(source: str) -> list[tuple[int, int, str, bool]]:
    """(start, end, text, strict) for every comment, docstring and message."""
    items: list[tuple[int, int, str, bool]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            body = token.string.lstrip('#').strip()
            if body and not body.startswith(DIRECTIVES):
                items.append((token.start[0], token.end[0], body, False))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=True)
            if doc:
                literal = node.body[0]
                items.append((literal.lineno, literal.end_lineno or literal.lineno,
                              doc, False))
        holders: list[ast.AST] = []
        if isinstance(node, (ast.Raise, ast.Assert)):
            holders = [node]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in LOG_METHODS):
            holders = list(node.args)
        for holder in holders:
            items.extend((start, end, text, True)
                         for start, end, text in _strict_strings(holder))
    return items


def markdown_prose(source: str) -> list[tuple[int, int, str, bool]]:
    """(line, line, text, False) for every Markdown line outside a fenced block."""
    items, fenced = [], False
    for number, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        # An indented block is code too, by Markdown's own four-space rule.
        if not fenced and line.strip() and not line.startswith('    '):
            items.append((number, number, line, False))
    return items


def json_prose(source: str) -> list[tuple[int, int, str, bool]]:
    """The operator-facing text in a schema: title, description, and x-notes."""
    texts: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('title', 'description') and isinstance(value, str):
                    texts.append(value)
                elif key == 'x-notes' and isinstance(value, list):
                    texts.extend(v for v in value if isinstance(v, str))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(source))
    items = []
    for text in texts:
        at = source.find(text)
        start = source[:at].count('\n') + 1 if at >= 0 else 0
        items.append((start, start + text.count('\n'), text, False))
    return items


EXTRACTORS = {'.py': python_prose, '.md': markdown_prose, '.json': json_prose}


def lint_file(path: Path) -> list[Finding]:
    """Every finding in one file, or none for a type this does not read."""
    extract = EXTRACTORS.get(path.suffix)
    if extract is None:
        return []
    name = path.as_posix()
    findings = []
    for start, end, text, strict in extract(path.read_text(encoding='utf-8')):
        for finding in check(name, start, text, strict):
            findings.append(replace(finding, end_line=end))
    return findings


def changed_lines(base: str) -> dict[str, set[int]]:
    """Line numbers the working tree and its commits have added since `base`."""
    diff = subprocess.run(['git', 'diff', '-U0', base], capture_output=True,
                          text=True, encoding='utf-8').stdout
    added: dict[str, set[int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith('+++ b/'):
            path = line[6:]
        elif line.startswith('@@') and path:
            match = re.search(r'\+(\d+)(?:,(\d+))?', line)
            start, count = int(match.group(1)), int(match.group(2) or 1)
            added.setdefault(path, set()).update(range(start, start + count))
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('paths', nargs='*', help='files to check')
    parser.add_argument('--changed', action='store_true',
                        help='check only the lines the diff against --base added')
    parser.add_argument('--base', default='HEAD',
                        help='what --changed compares against (default: HEAD)')
    args = parser.parse_args(argv)

    added: dict[str, set[int]] = {}
    paths = [Path(p) for p in args.paths]
    if args.changed:
        added = changed_lines(args.base)
        if not paths:
            paths = [Path(p) for p in added]

    findings = []
    for path in paths:
        if not path.exists():
            print(f'{path}: no such file', file=sys.stderr)
            continue
        for finding in lint_file(path):
            if added and not finding.touches(added.get(finding.path, set())):
                continue
            findings.append(finding)

    for finding in sorted(findings, key=lambda f: (f.path, f.line)):
        print(finding.render())
    if findings:
        print(f'\n{len(findings)} finding(s). See docs/ste-writing.md.')
        return 1
    print(f'clean: {len(paths)} file(s)')
    return 0


if __name__ == '__main__':                      # pragma: no cover -- CLI entry point
    sys.exit(main())
