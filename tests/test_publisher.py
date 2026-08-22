"""Tests for Publisher: index HTML generation and SCP upload logic."""

import errno
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from buzz.config import BuzzConfig
from buzz.publisher import CURRENT_CHART_NAME, Publisher


def _make_publisher(tmp_path: Path | None = None, current_chart: str = 'copy') -> Publisher:
    cfg = BuzzConfig()
    cfg.station.callsign = 'W6TST'
    cfg.station.timezone = 'America/Los_Angeles'
    cfg.audio.pulse_rate = 120
    cfg.server.host = 'testhost'
    cfg.server.username = 'testuser'
    cfg.server.remote_path = '/remote/'
    cfg.server.current_chart = current_chart
    if tmp_path:
        cfg.station.path = str(tmp_path)
    return Publisher(cfg)


class TestCurrentChartMode:
    def test_an_unknown_mode_is_refused_at_construction(self):
        """A typo in current_chart must fail loudly at startup rather than at upload.

        The upload path swallows every exception so a transient network fault cannot
        kill the monitor, which means a bad setting would otherwise be logged once a
        minute forever with the page silently never updating.
        """
        cfg = BuzzConfig()
        cfg.server.current_chart = 'symlnik'
        with pytest.raises(ValueError, match='symlnik'):
            Publisher(cfg)

    def test_both_supported_modes_construct(self):
        assert _make_publisher(current_chart='copy') is not None
        assert _make_publisher(current_chart='symlink') is not None


class TestGenerateIndex:
    def test_creates_html_file(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        assert output.exists()

    def test_callsign_in_output(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        assert 'W6TST' in output.read_text()

    def test_pulse_rate_in_output(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        assert '120' in output.read_text()

    def test_station_timezone_in_output(self, tmp_path):
        """The page compares station-local dates, so it needs the station's IANA zone.

        Without it the script would fall back to the reader's own zone and pause at
        whatever hour their midnight happens to be, rather than at the station's.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        assert 'America/Los_Angeles' in output.read_text()

    def test_page_points_at_the_fixed_chart_url(self, tmp_path):
        """The page must name only the fixed URL, never a dated chart.

        A dated filename baked into the page is the bug this whole design removes: it
        goes stale at the station's midnight and the page has no way to know.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        assert 'data/current.png' in content
        assert 'noise_plot' not in content

    def test_no_meta_refresh(self, tmp_path):
        """The script refreshes the image, so a whole-page reload must not also happen.

        A meta refresh would throw away the script's baseline date every minute, which
        is what decides whether the station has rolled over to a new day.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        assert 'http-equiv="refresh"' not in output.read_text()

    def test_the_page_fits_a_narrow_screen(self, tmp_path):
        """The chart is 1600 px wide and has to fit whatever screen it arrives on.

        Three things together made it not fit, and any one of them coming back brings
        the fault back: with no viewport tag a phone lays out at a notional desktop
        width, with no max-width the image draws at its full size regardless, and an
        overflowing flex item centered by justify-content spills off both edges with
        the left half in negative scroll space that cannot be scrolled to.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        assert 'name="viewport"' in content and 'width=device-width' in content, (
            'Without the viewport meta tag a phone lays the page out at about 980 px '
            'and scales it down.  The CSS below cannot then fit the real screen.'
        )
        assert 'max-width: 100%' in content, (
            'The chart must be allowed to shrink to the screen.  At its intrinsic 1600 '
            'px it overflows a centered flex container off both sides at once.'
        )
        assert 'width: 100vw' not in content, (
            'width: 100vw counts the scrollbar gutter, so it is wider than the space '
            'available and forces a horizontal scrollbar.'
        )
        # 'min-height: 100vh' contains 'height: 100vh', so this has to match the
        # declaration rather than the substring.
        assert not re.search(r'(?<!min-)(?<!max-)height:\s*100vh', content), (
            'A fixed viewport height cannot grow, so a portrait phone pushes content '
            'off the bottom.  Use min-height so the page extends instead.'
        )

    def test_every_link_and_source_is_relative_to_the_page(self, tmp_path):
        """The page must not assume which URL the station publishes it at.

        The archive link was written as /noise/data/, which is one station's own
        layout.  Anyone publishing to a different path got a link to a directory that
        does not exist, and only that station would ever see it work.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        assert 'href="/' not in content and 'src="/' not in content, (
            'An absolute path here binds the page to one site layout.  Relative URLs '
            'resolve correctly wherever the station publishes the page.'
        )
        assert 'href="data/"' in content

    def test_the_page_does_not_carry_a_timestamp(self, tmp_path):
        """The update time comes from Last-Modified, not from a value rendered here.

        A page cached by the browser would serve a rendered timestamp that is older
        than the chart beside it, which is worse than showing nothing.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        assert 'last-modified' in output.read_text().lower()

    def test_a_reader_without_javascript_is_told_what_the_page_cannot_do(self, tmp_path):
        """The status line promises what only the script can deliver, so it hides itself.

        Removing the <meta http-equiv="refresh"> moved the whole update mechanism into
        the script.  The minute-by-minute swap and the Last-Modified timestamp both
        need it, so with scripting off the sentence "This page updates once per minute"
        is simply false, and the reader has no way to tell the chart is stale.

        The two halves are separate elements and either could be dropped on its own,
        which is what this guards.  Hiding the line without the explanation leaves a
        reader with no status at all, and the explanation without the style rule leaves
        the page making both claims at once.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        head = content[:content.index('</head>')]
        assert re.search(r'<noscript>\s*<style[^>]*>\s*span#status\s*\{\s*display:\s*none',
                         head), (
            'The status line claims an update once per minute, which nothing performs '
            'with scripting off.  A noscript style rule in the head has to hide it.'
        )
        explanation = re.search(r'<noscript><span[^>]*>(.*?)</span></noscript>', content,
                                re.S)
        assert explanation is not None, (
            'Hiding the status line without replacing it leaves a reader no indication '
            'that the page has stopped updating itself.'
        )
        assert 'JavaScript' in explanation.group(1), (
            'The replacement has to name what is turned off, or a reader cannot act on '
            'it.  Found instead: ' + ' '.join(explanation.group(1).split())
        )

    def test_an_unknown_timezone_cannot_kill_the_script(self, tmp_path):
        """Intl.DateTimeFormat throws on a zone the browser's ICU build lacks.

        The formatter is built at the top level of the IIFE, before the first refresh
        and before the interval is set, so an unguarded throw there stops the whole
        script: no fetch, no image swap, and a status line still promising a minute-by-
        minute update.  Python's ZoneInfo accepts names, and backward links, that a
        trimmed ICU build does not, so config validation on this side is not a proof
        that the browser will agree.

        This reads the rendered page rather than running it, because the suite has no
        JavaScript engine.  It therefore pins the structure that makes the throw
        survivable, not the behavior itself.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        assert re.search(r'try\s*\{[^{}]*new Intl\.DateTimeFormat', content), (
            'new Intl.DateTimeFormat must be inside a try.  Unguarded at IIFE top '
            'level, a RangeError from an unknown timeZone ends the entire script.'
        )
        assert re.search(r'if\s*\(\s*stationDateParts === null\s*\)', content), (
            'stationDate has to answer for a formatter that was never built.  With '
            'the comparison disabled the page keeps updating and simply never pauses '
            'at midnight, which beats not updating at all.'
        )

    def test_the_page_gives_up_an_hour_after_the_last_new_chart(self, tmp_path):
        """Bounding the requests is the whole reason the page ever stops polling.

        The midnight pause only fires when a fetched Last-Modified carries a new
        station-local date, so a station that stops uploading at noon produces the same
        header forever and the page would poll once a minute for as long as it stayed
        open.  A reader who leaves a tab open over a weekend would make thousands of
        requests to learn nothing.

        The hour is expressed as a duration and divided by the poll interval rather
        than written as a count of 60, so changing how often the page polls cannot
        silently change how long it waits before giving up.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        refresh_ms = int(re.search(r'var REFRESH_MS = (\d+);', content).group(1))
        stale_ms = int(re.search(r'var STALE_AFTER_MS = (\d+);', content).group(1))
        assert stale_ms == 60 * 60 * 1000, (
            f'The page gives up after {stale_ms} ms.  One hour of unchanged polls was '
            'the agreed limit.'
        )
        assert re.search(r'var UNCHANGED_LIMIT = STALE_AFTER_MS / REFRESH_MS;', content), (
            'The poll count has to be derived from the two durations.  A literal 60 '
            'stays 60 when the refresh interval moves.  The page then gives up after '
            'the wrong span of time, and nothing here notices.'
        )
        assert stale_ms % refresh_ms == 0, (
            f'{stale_ms} ms does not divide by a {refresh_ms} ms poll.  The limit is '
            'then fractional, and no poll ever falls on it.'
        )

    def test_a_poll_that_brings_no_new_chart_counts_toward_the_stall(self, tmp_path):
        """Every way a cycle can fail has to count, or the page still polls forever.

        An unchanged Last-Modified is the expected case, but a server answering 503,
        a header that does not parse, and a dropped connection are all "no new chart"
        as well.  Counting only the tidy one would leave a broken server polled until
        the tab closes, which is the behavior this is here to stop.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        body = content[content.index('function refresh()'):]
        assert body.count('noNewChart();') == 4, (
            f'refresh() counts {body.count("noNewChart();")} of the four ways a poll '
            'can bring no chart: not ok, an unparseable date, an unchanged date, and '
            'a rejected fetch.'
        )
        assert re.search(r'\.catch\(function \(\) \{[^}]*noNewChart\(\);', body, re.S), (
            'A fetch that rejects has to count too.  A station whose server is down '
            'would otherwise be polled forever.'
        )

    def test_an_unchanged_chart_is_not_fetched_or_repainted(self, tmp_path):
        """A 304 means the bytes on screen are already right.

        Reading the body and swapping in a new object URL for identical bytes costs a
        decode a minute and gains nothing.  The early return also has to come before
        response.blob(), or the saving is only the repaint.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        body = content[content.index('function refresh()'):]
        guard = body.index('modified.getTime() === shownModified')
        assert guard < body.index('response.blob()'), (
            'The unchanged check must return before the body is read, or the page '
            'still downloads and decodes a chart it is already showing.'
        )

    def test_the_midnight_pause_does_not_depend_on_a_response_body(self, tmp_path):
        """response.body is null for some responses, and cancel() would then throw.

        The throw happens inside the then callback, where the outer catch swallows it,
        so showPaused never runs, the interval is never cleared, and the reader keeps
        polling a page frozen at yesterday's chart with nothing to say so.
        """
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        pub.generate_index(output)
        content = output.read_text()
        assert re.search(r'if\s*\(\s*response\.body\s*\)\s*\{\s*response\.body\.cancel\(\)',
                         content), (
            'response.body.cancel() has to be guarded.  A no-body response, or a '
            'revalidated cache hit served without a stream, makes it a TypeError that '
            'silently costs the reader the midnight pause.'
        )


class TestScpToServer:
    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    def _renamed_targets(self, mock_sftp) -> list[str]:
        """Remote paths that were renamed into place, in the order it happened."""
        return [c.args[1] for c in mock_sftp.method_calls if c[0] == 'posix_rename']

    def test_connects_to_configured_host(self, tmp_path):
        pub = _make_publisher(tmp_path)
        mock_client, _ = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([])
        mock_client.connect.assert_called_once_with(
            'testhost', username='testuser', password='', key_filename=pub._config.server.key_path
        )

    def test_rejects_unknown_host_keys(self, tmp_path):
        import paramiko
        pub = _make_publisher(tmp_path)
        mock_client, _ = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([])
        mock_client.load_system_host_keys.assert_called_once()
        policy = mock_client.set_missing_host_key_policy.call_args[0][0]
        assert isinstance(policy, paramiko.RejectPolicy)

    def test_puts_file_at_correct_remote_path(self, tmp_path):
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'data.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        assert self._renamed_targets(mock_sftp) == ['/remote/data/data.csv']

    def test_uploads_multiple_files(self, tmp_path):
        pub = _make_publisher(tmp_path)
        files = []
        for name in ('a.csv', 'b.png', 'c.html'):
            p = tmp_path / name
            p.write_text('x')
            files.append((p, 'data/'))
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server(files)
        assert mock_sftp.put.call_count == 3

    def test_sftp_closed_on_success(self, tmp_path):
        pub = _make_publisher(tmp_path)
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([])
        mock_sftp.close.assert_called_once()
        mock_client.close.assert_called_once()

    def test_connection_error_is_caught(self, tmp_path):
        pub = _make_publisher(tmp_path)
        mock_client = MagicMock()
        mock_client.connect.side_effect = Exception('connection refused')
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([])  # should not raise

    def test_client_closed_even_on_error(self, tmp_path):
        pub = _make_publisher(tmp_path)
        mock_client = MagicMock()
        mock_client.open_sftp.side_effect = Exception('sftp failed')
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([])
        mock_client.close.assert_called_once()


class TestTheRemotePathIsNormalized:
    """A remote path without its trailing slash silently misplaces every upload.

    /var/www/html/noise becomes /var/www/html/noiseindex.html, the SFTP put succeeds,
    nothing is logged, and the page is simply never where the web server looks.  The
    how-to guide's own example omitted the slash, so this is not a hypothetical typo.
    """

    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    @pytest.mark.parametrize('configured', ['/var/www/html/noise', '/var/www/html/noise/'])
    def test_uploads_go_to_the_same_place_with_or_without_the_slash(self, tmp_path, configured):
        pub = _make_publisher(tmp_path)
        pub._config.server.remote_path = configured
        pub = Publisher(pub._config)
        chart = tmp_path / 'noise_plot_movavg.2026-08-21.png'
        chart.write_text('png')
        index = tmp_path / 'index.html'
        index.write_text('<html></html>')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(chart, 'data/'), (index, '')], current_chart=chart)
        targets = [c.args[1] for c in mock_sftp.method_calls if c[0] == 'posix_rename']
        assert targets == [
            '/var/www/html/noise/data/noise_plot_movavg.2026-08-21.png',
            '/var/www/html/noise/index.html',
            f'/var/www/html/noise/data/{CURRENT_CHART_NAME}',
        ]

    def test_an_empty_remote_path_stays_empty(self, tmp_path):
        """An empty remote path means the SSH login directory, so it gains no slash.

        This is the dataclass default and a working configuration, not a disabled one.
        Adding a slash here would turn every target absolute and publish to the root of
        the server's filesystem instead.  See TestTheStagingPathAlwaysHasADirectory for
        what the resulting directory-less targets do to the staging name.
        """
        pub = _make_publisher(tmp_path)
        pub._config.server.remote_path = ''
        assert Publisher(pub._config)._remote_path == ''


class TestUploadsAreAtomic:
    """sftp.put() writes in place, so a reader can see a half-written file.

    Every upload therefore goes to a staging name and is renamed over its target.  The
    window matters most for current.png, which every viewer fetches once per minute.
    """

    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    def test_a_file_is_never_written_directly_to_its_final_path(self, tmp_path):
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'noise.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        written_to = mock_sftp.put.call_args[0][1]
        assert written_to != '/remote/data/noise.csv', (
            'put() wrote straight to the published path, so a browser fetching during '
            'the transfer would read a truncated file.'
        )
        assert written_to == '/remote/data/.uploading'
        mock_sftp.posix_rename.assert_called_once_with(written_to, '/remote/data/noise.csv')

    def test_a_cycle_never_has_two_files_staged_at_once(self, tmp_path):
        """One staging name is reused per directory, so each transfer has to be renamed
        away before the next one starts.

        That holds because a cycle uploads sequentially over a single connection, and
        the rename consumes the staging file.  It is what makes an abandoned staging
        file harmless, since the next upload simply overwrites it.  If uploads were
        ever made concurrent, two files would write to the same name and one would be
        published carrying the other's bytes, which no other test here would catch.
        """
        pub = _make_publisher(tmp_path, current_chart='copy')
        files = []
        for name in ('noise.csv', 'noise_plot.png', 'noise_plot_movavg.png'):
            p = tmp_path / name
            p.write_text('x')
            files.append((p, 'data/'))
        index = tmp_path / 'index.html'
        index.write_text('<html></html>')
        files.append((index, ''))
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server(files, current_chart=tmp_path / 'noise_plot_movavg.png')
        transfers = [c for c in mock_sftp.method_calls if c[0] in ('put', 'posix_rename')]
        kinds = [c[0] for c in transfers]
        assert kinds == ['put', 'posix_rename'] * (len(kinds) // 2), (
            f'Uploads must strictly alternate put then rename, but went {kinds}.  Two '
            'puts in a row means two files sharing one staging name.'
        )
        for staged, renamed in zip(transfers[::2], transfers[1::2]):
            assert staged.args[1].endswith('/.uploading')
            assert renamed.args[0] == staged.args[1]

    def test_falls_back_when_the_server_has_no_posix_rename(self, tmp_path):
        """posix_rename is an OpenSSH extension, not part of the SFTP protocol.

        Where it is missing, plain rename() refuses to overwrite, so the target has to
        go first.  Without this fallback every upload to such a server would fail.
        """
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'noise.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        mock_sftp.posix_rename.side_effect = IOError('unsupported')
        # Plain rename refuses the existing target once, which is what sends the
        # fallback on to remove it.  The retry afterwards succeeds.
        mock_sftp.rename.side_effect = [IOError('file exists'), None]
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        assert mock_sftp.remove.call_args_list[-1].args == ('/remote/data/noise.csv',)
        assert mock_sftp.rename.call_count == 2
        assert mock_sftp.rename.call_args[0][1] == '/remote/data/noise.csv'

    def test_a_missing_target_needs_no_removal_at_all(self, tmp_path):
        """The very first upload has nothing to replace, so the plain rename works.

        Trying the rename before removing anything is what makes this the cheap path
        rather than a remove that has to fail first.
        """
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'noise.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        mock_sftp.posix_rename.side_effect = IOError('unsupported')
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        mock_sftp.rename.assert_called_once()
        assert '/remote/data/noise.csv' not in [c.args[0] for c in mock_sftp.remove.call_args_list]


class TestTheStagingPathAlwaysHasADirectory:
    """A target with no directory part must not become a directory itself.

    rsplit('/', 1)[0] returns the whole string when there is no separator, so the
    target 'index.html' produced the staging path 'index.html/.uploading'.  That is a
    file used as a directory, so the put failed, and because scp_to_server wraps the
    whole cycle in one try, everything after it was skipped as well: current.png was
    never published and the page stopped updating, every cycle, with nothing but a
    logged exception to show for it.  An empty [server] remote_path is the dataclass
    default, and index.html is uploaded with no prefix, so this is the ordinary
    configuration rather than a corner case.
    """

    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    @pytest.mark.parametrize('target, expected', [
        ('index.html', './.uploading'),           # No directory at all: the login directory.
        ('/index.html', '/.uploading'),           # The server's root, which is a real path.
        ('data/current.png', 'data/.uploading'),
        ('/remote/data/current.png', '/remote/data/.uploading'),
    ])
    def test_the_staging_name_sits_beside_its_target(self, tmp_path, target, expected):
        assert _make_publisher(tmp_path)._staging_path_for(target) == expected, (
            f'The staging file for {target} must be a sibling of it.  Anywhere else '
            'and the rename crosses directories, which is no longer atomic, or names '
            'a path that cannot be written at all.'
        )

    def test_the_default_configuration_publishes_its_index(self, tmp_path):
        """The whole cycle with remote_path unset, which is how this was found."""
        pub = _make_publisher(tmp_path)
        pub._config.server.remote_path = ''
        pub = Publisher(pub._config)
        index = tmp_path / 'index.html'
        index.write_text('<html></html>')
        chart = tmp_path / 'noise_plot_movavg.2026-08-21.png'
        chart.write_text('png')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(chart, 'data/'), (index, '')], current_chart=chart)
        staged = [c.args[1] for c in mock_sftp.method_calls if c[0] == 'put']
        assert './.uploading' in staged, (
            f'index.html was staged at {staged}.  With no directory in the target the '
            'staging path has to fall back to the working directory.'
        )
        assert [c.args[1] for c in mock_sftp.method_calls if c[0] == 'posix_rename'] == [
            'data/noise_plot_movavg.2026-08-21.png',
            'index.html',
            f'data/{CURRENT_CHART_NAME}',
        ], 'The cycle must complete, including the current chart that commits it.'


class TestAFailedRenameLeavesThePublishedFileAlone:
    """The fallback removes the live file, so it must be sure of why it is doing that.

    paramiko raises IOError for every failed SFTP operation and gives an errno to only
    two of them, so a server without posix-rename and a read-only directory arrive
    looking identical.  Removing the target first meant the second case deleted a good
    chart and then failed to put anything in its place, so the page showed a broken
    image until some later cycle happened to work.  Doing nothing would have been
    better, which is the definition of a fallback that has overstepped.
    """

    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    def _upload(self, pub, tmp_path, mock_client):
        local = tmp_path / 'noise.csv'
        local.write_text('data')
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])

    def test_a_permission_error_deletes_nothing(self, tmp_path):
        pub = _make_publisher(tmp_path)
        mock_client, mock_sftp = self._mock_ssh()
        mock_sftp.posix_rename.side_effect = IOError(errno.EACCES, 'permission denied')
        self._upload(pub, tmp_path, mock_client)
        assert '/remote/data/noise.csv' not in [c.args[0] for c in mock_sftp.remove.call_args_list], (
            'A permission error is not a missing extension.  Removing the published '
            'file cannot succeed where the rename just failed, so it only destroys the '
            'copy readers currently have.'
        )
        mock_sftp.rename.assert_not_called()

    def test_a_missing_staging_file_deletes_nothing(self, tmp_path):
        """ENOENT here means the file being renamed is gone, not that the target is."""
        pub = _make_publisher(tmp_path)
        mock_client, mock_sftp = self._mock_ssh()
        mock_sftp.posix_rename.side_effect = IOError(errno.ENOENT, 'no such file')
        self._upload(pub, tmp_path, mock_client)
        assert '/remote/data/noise.csv' not in [c.args[0] for c in mock_sftp.remove.call_args_list]
        mock_sftp.rename.assert_not_called()

    def test_the_plain_rename_is_tried_before_anything_is_removed(self, tmp_path):
        """The ordering itself, since it is what makes the removal safe."""
        pub = _make_publisher(tmp_path)
        mock_client, mock_sftp = self._mock_ssh()
        mock_sftp.posix_rename.side_effect = IOError('unsupported')
        mock_sftp.rename.side_effect = [IOError('file exists'), None]
        self._upload(pub, tmp_path, mock_client)
        # The staging name is cleared before the put, which is a different concern.
        order = [c[0] for c in mock_sftp.method_calls
                 if c[0] == 'rename'
                 or (c[0] == 'remove' and c.args[0] != '/remote/data/.uploading')]
        assert order == ['rename', 'remove', 'rename'], (
            f'The fallback went {order}.  It has to attempt the rename first, because '
            'a rename that fails leaves the published file in place while a remove '
            'that succeeds does not.'
        )


class TestAnAbandonedStagingEntryCannotCorruptTheNextUpload:
    """A dropped connection can leave the staging name behind as a symlink.

    In symlink mode the staging name is created with sftp.symlink() and then renamed.
    If the connection drops in between, '.uploading' survives pointing at the day's
    dated chart.  put() writes through a symlink rather than replacing it, so the next
    cycle's CSV would be written into that chart, and the rename would then move the
    link itself over noise.csv.  One file corrupted, one replaced by a symlink, and
    nothing logged.
    """

    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    def test_the_staging_name_is_cleared_before_every_put(self, tmp_path):
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'noise.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        calls = [(c[0], c.args[0] if c.args else None) for c in mock_sftp.method_calls
                 if c[0] in ('remove', 'put')]
        assert calls[0] == ('remove', '/remote/data/.uploading'), (
            f'The first staging operation was {calls[0]}.  put() follows a symlink '
            'and writes through to its target.  Remove the staging name rather than '
            'overwriting it.'
        )

    def test_a_clean_staging_name_is_not_an_error(self, tmp_path):
        """The normal case: nothing staged, so the remove fails and is ignored."""
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'noise.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        mock_sftp.remove.side_effect = IOError('no such file')
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        mock_sftp.put.assert_called_once()
        mock_sftp.posix_rename.assert_called_once_with(
            '/remote/data/.uploading', '/remote/data/noise.csv')


class TestPublishingTheCurrentChart:
    """current.png is the one URL the page reads, and the commit point of a cycle."""

    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    def _renamed_targets(self, mock_sftp) -> list[str]:
        return [c.args[1] for c in mock_sftp.method_calls if c[0] == 'posix_rename']

    def test_copy_mode_uploads_the_chart_a_second_time(self, tmp_path):
        pub = _make_publisher(tmp_path, current_chart='copy')
        chart = tmp_path / 'noise_plot_movavg.2026-08-21.png'
        chart.write_text('png')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(chart, 'data/')], current_chart=chart)
        assert self._renamed_targets(mock_sftp) == [
            '/remote/data/noise_plot_movavg.2026-08-21.png',
            f'/remote/data/{CURRENT_CHART_NAME}',
        ]
        mock_sftp.symlink.assert_not_called()

    def test_symlink_mode_links_rather_than_uploading_twice(self, tmp_path):
        pub = _make_publisher(tmp_path, current_chart='symlink')
        chart = tmp_path / 'noise_plot_movavg.2026-08-21.png'
        chart.write_text('png')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(chart, 'data/')], current_chart=chart)
        # One put for the dated chart, and none for current.png.
        assert mock_sftp.put.call_count == 1
        target, staged_at = mock_sftp.symlink.call_args[0]
        assert target == 'noise_plot_movavg.2026-08-21.png', (
            'The link target must be a bare filename.  An absolute path would break if '
            'the published tree were ever moved to a different directory.'
        )
        assert staged_at == '/remote/data/.uploading'

    def test_the_current_chart_is_published_after_every_other_file(self, tmp_path):
        """Ordering is the whole reason no separate metadata file is needed.

        A browser polls current.png once per minute with no idea what else is being
        uploaded.  If it were published first, a poll arriving mid-cycle would be
        pointed at a chart still in transfer.
        """
        pub = _make_publisher(tmp_path, current_chart='copy')
        chart = tmp_path / 'noise_plot_movavg.2026-08-21.png'
        chart.write_text('png')
        csv = tmp_path / 'noise.2026-08-21.csv'
        csv.write_text('rows')
        index = tmp_path / 'index.html'
        index.write_text('<html></html>')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server(
                [(csv, 'data/'), (chart, 'data/'), (index, '')], current_chart=chart
            )
        targets = self._renamed_targets(mock_sftp)
        assert targets[-1] == f'/remote/data/{CURRENT_CHART_NAME}', (
            f'current.png must be the last thing published, but the order was {targets}.'
        )

    def test_no_current_chart_is_published_when_none_is_given(self, tmp_path):
        pub = _make_publisher(tmp_path, current_chart='copy')
        local = tmp_path / 'a.csv'
        local.write_text('x')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        assert CURRENT_CHART_NAME not in ' '.join(self._renamed_targets(mock_sftp))
