"""Tests for Publisher: index HTML generation and SCP upload logic."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

import pytest

from buzz.config import BuzzConfig
from buzz.publisher import Publisher

_TZ = ZoneInfo('America/Los_Angeles')


def _make_publisher(tmp_path: Path | None = None) -> Publisher:
    cfg = BuzzConfig()
    cfg.station.callsign = 'W6TST'
    cfg.audio.pulse_rate = 120
    cfg.server.host = 'testhost'
    cfg.server.username = 'testuser'
    cfg.server.remote_path = '/remote/'
    if tmp_path:
        cfg.station.path = str(tmp_path)
    return Publisher(cfg)


class TestGenerateIndex:
    def test_creates_html_file(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        when = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        pub.generate_index(output, when, 'data/plot.png')
        assert output.exists()

    def test_callsign_in_output(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        when = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        pub.generate_index(output, when, 'data/plot.png')
        assert 'W6TST' in output.read_text()

    def test_pulse_rate_in_output(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        when = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        pub.generate_index(output, when, 'data/plot.png')
        assert '120' in output.read_text()

    def test_image_path_in_output(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        when = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        pub.generate_index(output, when, 'data/noise_plot.2024-01-15.png')
        assert 'data/noise_plot.2024-01-15.png' in output.read_text()

    def test_no_meta_refresh_at_end_of_day(self, tmp_path):
        from datetime import time
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        # 23:59 with tzinfo triggers no_refresh=True
        when = datetime(2024, 1, 15, 23, 59, 0, tzinfo=_TZ)
        pub.generate_index(output, when, 'data/plot.png')
        content = output.read_text()
        assert 'meta http-equiv="refresh"' not in content

    def test_meta_refresh_present_during_day(self, tmp_path):
        pub = _make_publisher(tmp_path)
        output = tmp_path / 'index.html'
        when = datetime(2024, 1, 15, 10, 30, 0, tzinfo=_TZ)
        pub.generate_index(output, when, 'data/plot.png')
        content = output.read_text()
        assert 'refresh' in content


class TestScpToServer:
    def _mock_ssh(self):
        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        return mock_client, mock_sftp

    def test_connects_to_configured_host(self, tmp_path):
        pub = _make_publisher(tmp_path)
        mock_client, _ = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([])
        mock_client.connect.assert_called_once_with(
            'testhost', username='testuser', password='', key_filename=pub._config.server.key_path
        )

    def test_puts_file_at_correct_remote_path(self, tmp_path):
        pub = _make_publisher(tmp_path)
        local = tmp_path / 'data.csv'
        local.write_text('data')
        mock_client, mock_sftp = self._mock_ssh()
        with patch('buzz.publisher.paramiko.SSHClient', return_value=mock_client):
            pub.scp_to_server([(local, 'data/')])
        mock_sftp.put.assert_called_once_with(str(local), '/remote/data/data.csv')

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
