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


class Result:
    """What `subprocess.run` hands back, for the tests that stand in for git."""

    def __init__(self, stdout: str, code: int = 0, stderr: str = '') -> None:
        self.stdout, self.returncode, self.stderr = stdout, code, stderr


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

    def test_a_sentence_may_end_on_a_figure(self):
        """The lookbehind once rejected any digit, so a message that closed on a
        number ran into the next sentence.  Both strict checks then read the pair as
        one: the word cap counted the concatenation, and the trailing-preposition
        check only ever saw the last clause."""
        assert sentences('The gate is 97. Coverage fell.') == ['The gate is 97', 'Coverage fell.']

    def test_a_sentence_may_end_on_an_acronym(self):
        assert sentences('The scan uses an FFT. It is cheap.') == [
            'The scan uses an FFT', 'It is cheap.']

    def test_an_initial_is_not_a_sentence_end(self):
        """What the lookbehind is actually for: one capital letter and a period."""
        assert len(sentences('Written by N. Johnson today.')) == 1


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
        """CHANGELOG.md, CLAUDE.md and the schema have never used two spaces.

        docs/ste-writing.md is exempt on different grounds: it was adapted from an
        outside skill under the MIT License, so reflowing its 45 borrowed sentences
        would obscure which parts this project actually changed.
        """
        for path in ('CHANGELOG.md', 'CLAUDE.md', 'docs/ste-writing.md',
                     'lib/buzz/setup/schema.json'):
            assert 'one space after a period' not in rules(check(path, 1, 'One. Two.', False)), path

    def test_an_exempt_file_is_still_checked_for_everything_else(self):
        assert 'em dash' in rules(check('CHANGELOG.md', 1, 'a — b', False))

    def test_the_rule_book_passes_the_tool_that_enforces_it(self):
        """Every finding this file ever produced was one of its own examples, and a
        wall of them on the one document nobody may edit blind is worse than none."""
        rule_book = Path(__file__).resolve().parents[1] / 'docs' / 'ste-writing.md'
        found = lint_file(rule_book)
        assert not found, '\n'.join(f.render() for f in found)

    def test_an_abbreviation_is_not_a_sentence_end(self):
        """"e.g. Something" is one sentence, so the space after it is not the
        between-sentences space the rule is about."""
        assert 'one space after a period' not in rules(
            check('a.py', 1, 'a few rates, e.g. Nyquist at the floor', False))

    def test_an_ellipsis_is_not_a_sentence_end(self):
        assert 'one space after a period' not in rules(
            check('a.py', 1, 'and so on... Then it stops.', False))

    def test_a_sentence_that_closes_on_a_figure_is_still_checked(self):
        """The lookbehind that excluded a digit meant no violation after a number
        was ever reported, which is a whole class of them."""
        assert 'one space after a period' in rules(
            check('a.py', 1, 'The gate is 97. Coverage fell.', False))

    def test_a_numbered_step_is_not_a_sentence_end(self):
        """A numbered step opens its line with a figure and a period, and the space
        after that marker is not the between-sentences space the rule is about."""
        assert 'one space after a period' not in rules(
            check('a.md', 1, '1. Split any sentence over the cap.', False))


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

    def test_a_rule_book_may_quote_every_word_it_bans(self):
        """The spelling and wordiness rules are quoted in ste-writing.md in order to
        ban them, so the rule book tripped on its own examples: 20 findings on the one
        file a contributor may not edit blind."""
        text = '*synchronised* and *quantisation*, and *utilize* rather than *use*'
        assert not rules(check('docs/ste-writing.md', 1, text, False))
        assert rules(check('docs/other.md', 1, text, False)) == {'British spelling', 'wordy'}

    def test_a_rule_book_is_still_checked_for_punctuation(self):
        """The exemption covers vocabulary only.  A rule book is no freer to carry
        an em dash than any other file."""
        assert 'em dash' in rules(check('CLAUDE.md', 1, 'a — b', False))

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

    def test_an_assertions_test_expression_is_not_part_of_its_message(self):
        """Reading the whole statement glued the compared string onto the tail of the
        message, so the message no longer ended where the writer ended it and the
        trailing preposition this one carries went unreported."""
        source = "assert state == 'unlocked', 'the analyzer did not lock in the time given at'\n"
        assert [t for _, _, t, _ in python_prose(source)] == [
            'the analyzer did not lock in the time given at']
        assert 'ends with a preposition' in rules(
            check('a.py', 1, python_prose(source)[0][2], True))

    def test_an_assertion_with_no_message_yields_nothing(self):
        assert python_prose("assert state == 'the value it should have been'\n") == []

    def test_concatenated_pieces_keep_the_order_the_source_wrote_them(self):
        """ast.walk is breadth first, so it returned the halves of a join reversed.
        A message read backwards invents a fault at the seam and hides the real one."""
        source = ('raise RuntimeError(f"could not read {path} "\n'
                  '                   "and the fallback was empty")\n')
        assert [t for _, _, t, _ in python_prose(source)] == [
            'could not read {} and the fallback was empty']

    def test_an_interpolated_expression_is_code_not_prose(self):
        """A dictionary key inside `{}` is not a word of the sentence."""
        source = 'raise RuntimeError(f"could not read the {config[\'centre\']} setting")\n'
        assert [t for _, _, t, _ in python_prose(source)] == [
            'could not read the {} setting']


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

    def test_escaped_text_is_located_rather_than_reported_at_line_zero(self):
        """A description carrying a quote is stored escaped, so a search for the
        decoded form misses it.  The schema's own ssh-keygen note is one, and under
        --changed a line of 0 matches no added line, so edits to it went unchecked."""
        source = ('{\n "properties": {\n  "a": {\n'
                  r'   "description": "He said \"go\" to me."' + '\n}}}')
        items = json_prose(source)
        assert [t for _, _, t, _ in items] == ['He said "go" to me.']
        assert items[0][0] == 4, 'the escaped form is found where the file wrote it'

    def test_two_identical_descriptions_get_their_own_lines(self):
        """Searching from the start every time put both on the first one's line."""
        source = ('{"properties": {\n "a": {"description": "The same words here."},\n'
                  ' "b": {"description": "The same words here."}}}')
        assert [line for line, _, _, _ in json_prose(source)] == [2, 3]

    def test_text_that_cannot_be_located_is_reported_rather_than_dropped(self):
        """A file is free to escape more than JSON requires, and a unicode escape
        standing in for a plain letter defeats both forms of the search.  Line 0 is
        the honest answer, and Finding.touches then reports the item rather than
        filtering it away."""
        escaped_capital_a = chr(92) + 'u0041'
        source = '{"a": {"description": "' + escaped_capital_a + ' stale line of prose."}}'
        items = json_prose(source)
        assert [t for _, _, t, _ in items] == ['A stale line of prose.']
        assert items[0][0] == 0
        assert Finding('s.json', 0, 'rule', 'detail', 'text', False).touches(set()), (
            'A finding with no line number must survive --changed.  Dropping it is '
            'the silent miss the tool exists to prevent.')

    def test_the_whole_schema_is_located(self):
        """Nothing in the file this tool actually checks may fall back to line 0."""
        schema = Path(__file__).resolve().parents[1] / 'lib' / 'buzz' / 'setup' / 'schema.json'
        unlocated = [t for line, _, t, _ in json_prose(schema.read_text(encoding='utf-8'))
                     if line == 0]
        assert not unlocated, f'no line number found for: {unlocated}'


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

    def test_a_released_changelog_section_is_left_alone(self, tmp_path):
        """ste-writing.md override 3: a released section describes what shipped, and
        the published release notes were lifted from it verbatim.  Rewording it now
        would make the two disagree, so only [Unreleased] is checked."""
        path = tmp_path / 'CHANGELOG.md'
        path.write_text('## [Unreleased]\nThe colour scale moved.\n'
                        '## [1.5.1] - 2026-08-01\nThe colour scale moved.\n', encoding='utf-8')
        found = lint_file(path)
        assert [f.line for f in found] == [2], 'only the unreleased entry is reported'

    def test_a_changelog_with_no_releases_is_checked_throughout(self, tmp_path):
        path = tmp_path / 'CHANGELOG.md'
        path.write_text('## [Unreleased]\nThe colour scale moved.\n', encoding='utf-8')
        assert [f.line for f in lint_file(path)] == [2]


class TestChangedLines:
    def test_parses_added_line_numbers_from_a_diff(self, monkeypatch):
        diff = ('+++ b/one.py\n@@ -1,0 +2,3 @@\n+a\n+b\n+c\n'
                '+++ b/two.md\n@@ -5 +9 @@\n+x\n')
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: Result(diff))
        assert changed_lines('HEAD') == {'one.py': {2, 3, 4}, 'two.md': {9}}

    def test_a_failed_diff_raises_rather_than_reporting_nothing(self, monkeypatch):
        """git's failure gives an empty stdout, which parses as "nothing changed".
        A mistyped --base would then print "clean" and exit 0, which is the one
        answer a pre-commit gate must never give by accident."""
        monkeypatch.setattr(subprocess, 'run',
                            lambda *a, **k: Result('', code=128, stderr='fatal: bad revision'))
        with pytest.raises(RuntimeError) as failure:
            changed_lines('nosuchbranch')
        assert 'nosuchbranch' in str(failure.value) and 'bad revision' in str(failure.value)


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
        diff = f'+++ b/{path.as_posix()}\n@@ -1 +2 @@\n+x\n'
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: Result(diff))
        assert main(['--changed', str(path)]) == 1
        out = capsys.readouterr().out
        assert ':2:' in out and ':1:' not in out

    def test_changed_with_no_paths_takes_them_from_the_diff(self, tmp_path, monkeypatch, capsys):
        """The everyday invocation: `--changed` alone, checking whatever was edited."""
        path = tmp_path / 'edited.py'
        path.write_text('# One. Two.\n', encoding='utf-8')
        diff = f'+++ b/{path.as_posix()}\n@@ -0,0 +1 @@\n+x\n'
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: Result(diff))
        assert main(['--changed']) == 1
        assert 'one space after a period' in capsys.readouterr().out

    def test_a_bad_base_ref_is_named_and_does_not_report_clean(self, monkeypatch, capsys):
        """Exit 2 rather than 0, and no "clean" line: the run did not happen."""
        monkeypatch.setattr(subprocess, 'run',
                            lambda *a, **k: Result('', code=128, stderr='fatal: bad revision'))
        assert main(['--changed', '--base', 'nosuchbranch']) == 2
        out = capsys.readouterr()
        assert 'clean' not in out.out
        assert 'nosuchbranch' in out.err and '--base' in out.err

    def test_it_runs_as_a_script(self):
        """Exercises the argparse wiring the way a contributor invokes it."""
        result = subprocess.run([sys.executable, str(TOOL), str(TOOL)],
                                capture_output=True, text=True, encoding='utf-8')
        assert result.returncode == 0, result.stdout + result.stderr
