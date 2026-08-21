"""Tests for Publisher: index HTML generation and SCP upload logic."""

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
        """Uploads are off in this case, so nothing should get a leading slash."""
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
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        mock_sftp.remove.assert_called_once_with('/remote/data/noise.csv')
        mock_sftp.rename.assert_called_once()
        assert mock_sftp.rename.call_args[0][1] == '/remote/data/noise.csv'

    def test_a_missing_target_does_not_stop_the_fallback(self, tmp_path):
        """The very first upload has nothing to remove, and must still go through."""
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'noise.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        mock_sftp.posix_rename.side_effect = IOError('unsupported')
        mock_sftp.remove.side_effect = IOError('no such file')
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        mock_sftp.rename.assert_called_once()


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
