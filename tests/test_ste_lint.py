"""Tests for tools/ste_lint.py, the prose checker.

A linter that reports a fault which is not there is worse than no linter, because
the fix is to edit correct prose until the tool stops complaining.  So most of what
follows pins the cases that must *not* be reported: a decimal is not a sentence end,
`self.attribute` is not a sentence end, a paragraph break is not a missing space, and
an f-string's pieces are one sentence rather than several.

Each of those was a real defect in this tool before it earned its place in tools/.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ste_lint import (
    Finding, changed_lines, check, json_prose, lint_file, main, markdown_prose,
    python_prose, sentences,
)

TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'ste_lint.py'


def rules(findings) -> set[str]:
    return {f.rule for f in findings}


class TestSentences:
    def test_splits_on_a_period(self):
        assert sentences('One thing. Another thing.') == ['One thing', 'Another thing.']

    def test_a_decimal_is_not_a_sentence_end(self):
        """"2.5 ms/div" appears throughout this codebase's prose."""
        assert len(sentences('The step is 2.5 ms and holds.')) == 1

    def test_a_dotted_name_is_not_a_sentence_end(self):
        assert len(sentences('Look at config.py for the band.')) == 1

    def test_collapses_wrapped_lines(self):
        assert sentences('A sentence\n    wrapped over lines.') == ['A sentence wrapped over lines.']


class TestSpacing:
    def test_one_space_is_reported(self):
        assert 'one space after a period' in rules(check('a.py', 1, 'One. Two.', False))

    def test_two_spaces_pass(self):
        assert 'one space after a period' not in rules(check('a.py', 1, 'One.  Two.', False))

    def test_a_newline_is_a_paragraph_break_not_a_missing_space(self):
        """The check was inverted once and flagged every paragraph in every docstring."""
        assert 'one space after a period' not in rules(check('a.py', 1, 'One.\nTwo.', False))

    def test_a_lowercase_continuation_is_not_a_sentence_end(self):
        assert 'one space after a period' not in rules(check('a.py', 1, 'see config.py for it', False))

    def test_exempt_files_are_not_checked_for_spacing(self):
        """CHANGELOG.md, CLAUDE.md and the schema have never used two spaces."""
        for path in ('CHANGELOG.md', 'CLAUDE.md', 'lib/buzz/setup/schema.json'):
            assert 'one space after a period' not in rules(check(path, 1, 'One. Two.', False)), path

    def test_an_exempt_file_is_still_checked_for_everything_else(self):
        assert 'em dash' in rules(check('CHANGELOG.md', 1, 'a — b', False))

    def test_an_abbreviation_is_not_a_sentence_end(self):
        """"e.g. Something" is one sentence, so the space after it is not the
        between-sentences space the rule is about."""
        assert 'one space after a period' not in rules(
            check('a.py', 1, 'a few rates, e.g. Nyquist at the floor', False))

    def test_an_ellipsis_is_not_a_sentence_end(self):
        assert 'one space after a period' not in rules(
            check('a.py', 1, 'and so on... Then it stops.', False))


class TestBannedWords:
    @pytest.mark.parametrize('word', ['genuine', 'genuinely', 'load-bearing', 'lands', 'landed'])
    def test_each_banned_word_is_caught(self, word):
        assert 'banned word' in rules(check('a.py', 1, f'This {word} thing', False))

    def test_a_rule_book_may_quote_the_words_it_bans(self):
        """CLAUDE.md lists them in order to ban them, and must not trip on its own list."""
        assert 'banned word' not in rules(check('CLAUDE.md', 1, 'These words: genuine, lands.', False))

    def test_marketing_adjectives(self):
        assert 'marketing adjective' in rules(check('a.py', 1, 'A robust design', False))

    def test_british_spellings(self):
        assert 'British spelling' in rules(check('a.py', 1, 'the centre of it', False))

    def test_a_rule_book_is_still_checked_for_spelling(self):
        assert 'British spelling' in rules(check('CLAUDE.md', 1, 'the colour of it', False))

    def test_wordy_choices_name_the_replacement(self):
        found = [f for f in check('a.py', 1, 'utilize the buffer', False) if f.rule == 'wordy']
        assert found and '"use"' in found[0].detail


class TestDashes:
    def test_em_dash_is_caught(self):
        assert 'em dash' in rules(check('a.py', 1, 'a — b', False))

    def test_a_spaced_en_dash_is_caught(self):
        assert 'en dash' in rules(check('a.py', 1, 'a – b', False))

    def test_a_numeric_range_is_typography_not_a_dash(self):
        """"2.5-6 ms" and "S1-S9" are ranges, and appear as en dashes throughout."""
        assert 'en dash' not in rules(check('a.py', 1, 'the 8–48 kHz band', False))
        assert 'en dash' not in rules(check('a.py', 1, 'S1–S9 green', False))

    def test_a_spaced_hyphen_is_the_house_style_and_passes(self):
        assert not rules(check('a.py', 1, 'a field tool - not lab gear', False))


class TestStrictOnly:
    def test_a_contraction_is_caught_in_strict_text(self):
        assert 'contraction' in rules(check('a.py', 1, "it isn't there", True))

    def test_a_contraction_is_allowed_in_a_comment(self):
        assert 'contraction' not in rules(check('a.py', 1, "it isn't there", False))

    def test_a_semicolon_is_caught_in_strict_text(self):
        assert 'semicolon' in rules(check('a.py', 1, 'it failed; try again', True))

    def test_a_semicolon_is_allowed_in_a_comment(self):
        """ste-writing.md override 1: banned in strict mode only."""
        assert 'semicolon' not in rules(check('a.py', 1, 'it failed; try again', False))

    def test_the_word_cap_applies_to_strict_text(self):
        long = ' '.join(['word'] * 30) + '.'
        assert 'over the strict cap' in rules(check('a.py', 1, long, True))

    def test_the_word_cap_does_not_apply_to_a_docstring(self):
        """ste-writing.md override 2: the caps are strict mode only."""
        long = ' '.join(['word'] * 30) + '.'
        assert 'over the strict cap' not in rules(check('a.py', 1, long, False))

    def test_a_trailing_preposition_is_caught(self):
        assert 'ends with a preposition' in rules(
            check('a.py', 1, 'the figure it was recorded at.', True))

    def test_the_recommended_rewrite_passes(self):
        assert 'ends with a preposition' not in rules(
            check('a.py', 1, 'the figure at which it was recorded.', True))


class TestPythonProse:
    def test_a_comment_is_flavored_prose(self):
        items = python_prose('# a note here\nx = 1\n')
        assert (1, 1, 'a note here', False) in items

    def test_a_tool_directive_is_not_prose(self):
        assert python_prose('x = 1  # noqa: E501\n') == []

    def test_a_docstring_carries_its_whole_span(self):
        """--changed needs the span, or an edit inside a docstring goes unseen."""
        source = 'def f():\n    """Line one.\n\n    Line four.\n    """\n'
        docs = [i for i in python_prose(source) if 'Line one' in i[2]]
        assert docs and docs[0][0] == 2 and docs[0][1] == 5

    def test_an_assertion_message_is_strict(self):
        items = python_prose("assert x, 'this message is here'\n")
        assert any(text == 'this message is here' and strict for _, _, text, strict in items)

    def test_a_raise_message_is_strict(self):
        items = python_prose("raise ValueError('the thing was not found')\n")
        assert any(strict for _, _, _, strict in items)

    def test_a_log_call_is_strict(self):
        items = python_prose("logger.warning('the device went away')\n")
        assert any(strict for _, _, _, strict in items)

    def test_an_fstring_is_rejoined_into_one_sentence(self):
        """Split into pieces, "started ffmpeg at {path}" looks like it ends in
        a preposition, and the tool invents a fault that is not there."""
        items = python_prose("raise RuntimeError(f'could not start ffmpeg at {path}')\n")
        text = [t for _, _, t, _ in items][0]
        assert text == 'could not start ffmpeg at {}'
        assert 'ends with a preposition' not in rules(check('a.py', 1, text, True))

    def test_a_short_label_is_not_treated_as_prose(self):
        assert python_prose("raise ValueError('too small')\n") == []

    def test_a_plain_string_is_not_strict_text(self):
        """Only a raise, an assert or a log call is what a stuck reader meets."""
        assert python_prose("message = 'this is just a value somewhere'\n") == []


class TestMarkdownProse:
    def test_a_paragraph_is_prose(self):
        assert markdown_prose('Some words here.\n') == [(1, 1, 'Some words here.', False)]

    def test_a_fenced_block_is_code(self):
        assert markdown_prose('```\nrm -rf /\n```\n') == []

    def test_an_indented_block_is_code(self):
        assert markdown_prose('    python -m buzz.main\n') == []

    def test_a_blank_line_is_skipped(self):
        assert markdown_prose('\n\n') == []


class TestJsonProse:
    SCHEMA = '{"properties": {"a": {"title": "A title", "description": "A description.",\n' \
             ' "x-notes": ["A note."], "default": 3}}}'

    def test_reads_operator_facing_text(self):
        texts = {t for _, _, t, _ in json_prose(self.SCHEMA)}
        assert texts == {'A title', 'A description.', 'A note.'}

    def test_ignores_values_that_are_not_prose(self):
        assert all('3' != t for _, _, t, _ in json_prose(self.SCHEMA))

    def test_it_walks_into_a_list_that_is_not_x_notes(self):
        """A nullable field writes its type as a list, and an enum's choices are one
        too.  Both have to be walked through to reach the descriptions below them."""
        source = ('{"properties": {"a": {"type": ["integer", "null"], '
                  '"anyOf": [{"description": "Buried under a list."}]}}}')
        assert [t for _, _, t, _ in json_prose(source)] == ['Buried under a list.']

    def test_text_that_is_escaped_in_the_source_still_gets_extracted(self):
        """A description carrying a quote is stored escaped, so it cannot be found
        literally in the file and gets no line number.  It is still prose an operator
        reads, so it is extracted and checked rather than dropped."""
        source = r'{"properties": {"a": {"description": "He said \"go\" to me."}}}'
        items = json_prose(source)
        assert [t for _, _, t, _ in items] == ['He said "go" to me.']
        assert items[0][0] == 0, 'text that cannot be located reports line 0'


class TestFindingSpan:
    def test_touches_any_line_of_its_prose(self):
        finding = Finding('a.py', 10, 'rule', 'detail', 'text', False, end_line=14)
        assert finding.touches({12})
        assert not finding.touches({20})

    def test_a_single_line_finding_needs_that_line(self):
        finding = Finding('a.py', 10, 'rule', 'detail', 'text', False)
        assert finding.touches({10})
        assert not finding.touches({11})


class TestLintFile:
    def test_an_unknown_suffix_is_skipped(self, tmp_path):
        path = tmp_path / 'notes.rst'
        path.write_text('One. Two.', encoding='utf-8')
        assert lint_file(path) == []

    def test_a_clean_file_reports_nothing(self, tmp_path):
        path = tmp_path / 'clean.py'
        path.write_text('# a plain note\nx = 1\n', encoding='utf-8')
        assert lint_file(path) == []

    def test_findings_carry_the_path_and_line(self, tmp_path):
        path = tmp_path / 'dirty.py'
        path.write_text('# One. Two.\n', encoding='utf-8')
        found = lint_file(path)
        assert found[0].line == 1 and found[0].path.endswith('dirty.py')

    def test_the_tool_passes_its_own_check(self):
        """It would be a poor advertisement otherwise."""
        assert lint_file(TOOL) == []


class TestChangedLines:
    def test_parses_added_line_numbers_from_a_diff(self, monkeypatch):
        diff = ('+++ b/one.py\n@@ -1,0 +2,3 @@\n+a\n+b\n+c\n'
                '+++ b/two.md\n@@ -5 +9 @@\n+x\n')

        class Result:
            stdout = diff

        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: Result())
        assert changed_lines('HEAD') == {'one.py': {2, 3, 4}, 'two.md': {9}}


class TestMain:
    def test_a_clean_file_exits_zero(self, tmp_path, capsys):
        path = tmp_path / 'clean.py'
        path.write_text('# a plain note\n', encoding='utf-8')
        assert main([str(path)]) == 0
        assert 'clean' in capsys.readouterr().out

    def test_a_finding_exits_one_and_is_printed(self, tmp_path, capsys):
        path = tmp_path / 'dirty.py'
        path.write_text('# One. Two.\n', encoding='utf-8')
        assert main([str(path)]) == 1
        assert 'one space after a period' in capsys.readouterr().out

    def test_a_missing_file_is_named_rather_than_crashing(self, tmp_path, capsys):
        assert main([str(tmp_path / 'gone.py')]) == 0
        assert 'no such file' in capsys.readouterr().err

    def test_changed_restricts_to_added_lines(self, tmp_path, monkeypatch, capsys):
        """The whole point of --changed: an old fault on an untouched line is not
        this change's to fix, and reporting it buries the one that is."""
        path = tmp_path / 'mixed.py'
        path.write_text('# Old. Fault.\n# New. Fault.\n', encoding='utf-8')

        class Result:
            stdout = f'+++ b/{path.as_posix()}\n@@ -1 +2 @@\n+x\n'

        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: Result())
        assert main(['--changed', str(path)]) == 1
        out = capsys.readouterr().out
        assert ':2:' in out and ':1:' not in out

    def test_changed_with_no_paths_takes_them_from_the_diff(self, tmp_path, monkeypatch, capsys):
        """The everyday invocation: `--changed` alone, checking whatever was edited."""
        path = tmp_path / 'edited.py'
        path.write_text('# One. Two.\n', encoding='utf-8')

        class Result:
            stdout = f'+++ b/{path.as_posix()}\n@@ -0,0 +1 @@\n+x\n'

        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: Result())
        assert main(['--changed']) == 1
        assert 'one space after a period' in capsys.readouterr().out

    def test_it_runs_as_a_script(self):
        """Exercises the argparse wiring the way a contributor invokes it."""
        result = subprocess.run([sys.executable, str(TOOL), str(TOOL)],
                                capture_output=True, text=True, encoding='utf-8')
        assert result.returncode == 0, result.stdout + result.stderr
