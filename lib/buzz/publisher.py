"""
HTML index generation and SCP upload to the web server.

Publisher.generate_index() renders index.html from the Jinja2 template, embedding
the current plot filename, timestamp, and station callsign.  Publisher.scp_to_server()
opens a single SSH connection and uploads a list of (local_path, remote_prefix) pairs
over SFTP.
"""

import logging
from datetime import datetime, time
from pathlib import Path

import jinja2
import paramiko
from jinja2 import FileSystemLoader

from buzz.config import BuzzConfig

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / 'templates'


class Publisher:
    def __init__(self, config: BuzzConfig) -> None:
        self._config = config
        environment = jinja2.Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
        self._template = environment.get_template('index.html')

    def generate_index(self, output_filename: Path | str, collection_time: datetime, image_path: str) -> None:
        collection_time_formatted = collection_time.strftime('%d %B %Y %H:%M:%S %Z (%z)')
        # The 23:59 collection is the last one of the calendar day.  Suppress the
        # auto-refresh on that final page so the browser doesn't reload after midnight
        # into the new day's (still empty) graph.
        no_refresh = collection_time.timetz() == time(23, 59, 0, 0, tzinfo=collection_time.tzinfo)
        content = self._template.render(
            filename=image_path,
            update_datetime=collection_time_formatted,
            no_refresh=no_refresh,
            callsign=self._config.station.callsign,
            pulse_rate=self._config.audio.pulse_rate,
        )
        with open(output_filename, mode='w', encoding='utf-8') as f:
            f.write(content)

    def scp_to_server(self, files: list[tuple[Path | str, str]]) -> None:
        """Upload files over a single SSH connection.

        Each entry in *files* is a (local_path, remote_prefix) pair; the file
        is placed at server_remote_path + remote_prefix + basename.
        """
        server = self._config.server
        sftp = None
        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(server.host, username=server.username,
                           password='', key_filename=server.key_path)
            sftp = client.open_sftp()
            for local_file, file_prefix in files:
                destination_name = Path(local_file).name
                sftp.put(str(local_file), f'{server.remote_path}{file_prefix}{destination_name}')
        except Exception:
            logger.exception(
                'Uploading output files to %s failed — check SSH key (%s), '
                'remote path (%s), and host reachability. '
                'Files will be re-uploaded next cycle.',
                server.host, server.key_path, server.remote_path,
            )
        finally:
            if sftp:
                sftp.close()
            if client:
                client.close()
