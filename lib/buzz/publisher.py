"""
HTML index generation and SCP upload to the web server.

Publisher.generate_index() renders index.html from the Jinja2 template.  The page is
static: it names no chart file and carries no timestamp, because the browser reads both
from the fixed `data/current.png` URL and its Last-Modified header.  Only the callsign,
the pulse rate, and the station timezone are substituted, so the rendered page changes
only when the config does.

Publisher.scp_to_server() opens a single SSH connection, uploads a list of
(local_path, remote_prefix) pairs over SFTP, and publishes `data/current.png` last.
Every write goes to a staging name and is renamed over its target, so a browser
fetching mid-upload cannot read a half-written file.
"""

import errno
import logging
from pathlib import Path

import jinja2
import paramiko
from jinja2 import FileSystemLoader

from buzz.config import BuzzConfig

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / 'templates'

# The fixed name the web page fetches every minute, relative to the remote path.  The
# dated charts keep their own names for the browsable archive; this is the one URL that
# never changes, which is what lets the page hold no date logic of its own.
CURRENT_CHART_NAME = 'current.png'
CURRENT_CHART_URL = f'data/{CURRENT_CHART_NAME}'

# Ways of publishing that fixed name.  'copy' re-uploads the chart under it and works
# on any server.  'symlink' points it at the dated chart instead, which saves the
# upload but needs the web server to follow symlinks.
COPY = 'copy'
SYMLINK = 'symlink'
_CURRENT_CHART_MODES = (COPY, SYMLINK)

# Every upload is written here first and then renamed over its target.  One reused name
# per directory rather than one per file, which is what makes an abandoned staging file
# harmless: the next upload into that directory overwrites it and renames it away.  A
# per-file name would strand the dated ones forever, since tomorrow's staging name
# carries tomorrow's date and nothing would ever touch today's again.  The leading dot
# also keeps it out of the web server's directory listing while a transfer is running.
_STAGING_NAME = '.uploading'


class Publisher:
    def __init__(self, config: BuzzConfig) -> None:
        self._config = config
        mode = config.server.current_chart
        if mode not in _CURRENT_CHART_MODES:
            raise ValueError(
                f'The [server] current_chart setting is "{mode}", and the only values '
                f'this program understands are "{COPY}" and "{SYMLINK}".  Check '
                f'~/.buzz/config.toml for a typo.  Use "{COPY}" unless the web server '
                'is known to follow symbolic links.')
        # Without its trailing slash a remote path concatenates straight onto the first
        # filename, so /var/www/html/noise becomes /var/www/html/noiseindex.html and
        # every upload goes somewhere nobody is serving.  Nothing fails: the SFTP put
        # succeeds, the log stays quiet, and the web page is simply never there.  The
        # sample config asked for the slash while the how-to guide's example omitted
        # it, so forgetting it is made harmless here rather than merely documented.
        self._remote_path = config.server.remote_path
        if self._remote_path and not self._remote_path.endswith('/'):
            self._remote_path += '/'
        environment = jinja2.Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
        self._template = environment.get_template('index.html')

    def generate_index(self, output_filename: Path | str) -> None:
        """Render index.html from the template and write it to `output_filename`.

        The page takes no collection time.  It reads the Last-Modified header of the
        chart it fetches, which is the moment the bytes reached the web rather than
        a figure this program writes into the page and hopes stays true.
        """
        content = self._template.render(
            chart_url=CURRENT_CHART_URL,
            callsign=self._config.station.callsign,
            pulse_rate=self._config.audio.pulse_rate,
            station_timezone=self._config.station.timezone,
        )
        with open(output_filename, mode='w', encoding='utf-8') as f:
            f.write(content)

    def _staging_path_for(self, remote_path: str) -> str:
        """The staging name every upload into `remote_path`'s own directory shares.

        Keeping it in the target's directory means the rename is always within one
        filesystem, which is what makes it atomic.  A single publisher uploads
        sequentially over one connection, so there is nothing to collide with.

        rpartition rather than rsplit, because rsplit returns the whole string when it
        finds no separator.  A target with no directory at all is the ordinary case for
        index.html when [server] remote_path is empty, and it made the staging path
        `index.html/.uploading`, which is a file used as a directory.  The put failed
        every cycle and took the rest of the upload down with it.
        """
        directory, separator, _ = remote_path.rpartition('/')
        # An empty directory with a separator present is the server's root, which is a
        # real path; an empty one without is "wherever SSH put us", which is '.'.
        return f'{directory if separator else "."}/{_STAGING_NAME}'

    def _clear_staging(self, sftp: paramiko.SFTPClient, staging_path: str) -> None:
        """Remove whatever an interrupted cycle left under the staging name.

        A staging file is harmless, because put() overwrites it.  A staging *symlink*
        is not: put() follows it and writes through to whatever it points at, so a
        symlink abandoned by a dropped connection would send the next file's bytes into
        the dated chart it was pointing at, and then rename the link itself over the
        target.  Removing it first costs one round trip and cannot leave that state.
        """
        try:
            sftp.remove(staging_path)
        except IOError:
            pass  # Nothing staged, which is the normal case.

    def _rename_over(self, sftp: paramiko.SFTPClient, staging_path: str, remote_path: str) -> None:
        """Move `staging_path` onto `remote_path`, replacing whatever is there.

        posix_rename() is the atomic one and is what makes the staging dance worth
        doing, but it is an OpenSSH extension rather than part of the SFTP protocol.
        Where the server does not offer it, plain rename() will not overwrite, so the
        target has to be removed first.  That leaves a window of a millisecond or so
        where the URL 404s, which the page already survives by keeping its current
        image, and is still far better than serving a truncated PNG.

        The fallback tries the plain rename before removing anything, and that order is
        the point.  paramiko raises IOError for every failed operation and gives only
        "no such file" and "permission denied" an errno, so an unsupported extension is
        indistinguishable by class from a full disk or a read-only directory.  Removing
        first meant those failures deleted a good chart and then failed to replace it,
        leaving the page showing a broken image until a later cycle succeeded, which is
        worse than doing nothing at all.  Renaming first cannot: it either works, or it
        fails with the target still in place.
        """
        try:
            sftp.posix_rename(staging_path, remote_path)
            return
        except IOError as exc:
            # These two carry an errno, so they are known to be real failures rather
            # than a missing extension, and retrying by hand would only repeat them.
            if exc.errno in (errno.EACCES, errno.ENOENT):
                raise
            logger.debug('posix-rename failed (%s), falling back to rename.', exc)
        try:
            sftp.rename(staging_path, remote_path)
            return
        except IOError:
            # Expected on the first fallback of every cycle after the first: plain
            # rename refuses an existing target, which is exactly what a republish has.
            pass
        sftp.remove(remote_path)
        sftp.rename(staging_path, remote_path)

    def _put_atomic(self, sftp: paramiko.SFTPClient, local_path: Path | str, remote_path: str) -> None:
        """Upload a file so that no reader can see it half-written.

        sftp.put() writes in place.  For the tens of milliseconds a chart takes to
        transfer, a browser fetching that URL reads a truncated PNG and draws a broken
        image.  Uploading to a staging name and renaming means a reader gets either the
        previous chart or the complete new one.
        """
        staging_path = self._staging_path_for(remote_path)
        self._clear_staging(sftp, staging_path)
        sftp.put(str(local_path), staging_path)
        self._rename_over(sftp, staging_path, remote_path)

    def _publish_current_chart(self, sftp: paramiko.SFTPClient, chart: Path | str, data_path: str) -> None:
        """Point the page's fixed URL at this cycle's chart, by copy or by symlink."""
        remote_path = f'{data_path}{CURRENT_CHART_NAME}'
        if self._config.server.current_chart == COPY:
            self._put_atomic(sftp, chart, remote_path)
            return
        staging_path = self._staging_path_for(remote_path)
        # symlink() fails outright if the name exists, and the staging name is reused,
        # so the previous cycle's copy has to go first.
        self._clear_staging(sftp, staging_path)
        # A relative target resolves within the data directory, so the link survives the
        # whole tree being moved to a different path on the server.
        sftp.symlink(Path(chart).name, staging_path)
        self._rename_over(sftp, staging_path, remote_path)

    def scp_to_server(self, files: list[tuple[Path | str, str]],
                      current_chart: Path | str | None = None) -> None:
        """Upload files over a single SSH connection.

        Each entry in *files* is a (local_path, remote_prefix) pair.  The file
        is placed at server_remote_path + remote_prefix + basename.

        *current_chart* is the dated chart the page's fixed URL should show.  It is
        published last, after every file in *files* has arrived, because it is the
        commit point: a browser polling in between must never be pointed at a chart
        that has not finished uploading.
        """
        server = self._config.server
        sftp = None
        client = None
        try:
            client = paramiko.SSHClient()
            # Verify the server's host key against known_hosts rather than trusting
            # whatever answers (AutoAddPolicy would accept a man-in-the-middle).
            client.load_system_host_keys()
            known_hosts = Path.home() / '.buzz' / 'known_hosts'
            if known_hosts.exists():
                client.load_host_keys(str(known_hosts))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(server.host, username=server.username,
                           password='', key_filename=server.key_path)
            sftp = client.open_sftp()
            for local_file, file_prefix in files:
                destination_name = Path(local_file).name
                self._put_atomic(sftp, local_file, f'{self._remote_path}{file_prefix}{destination_name}')
            if current_chart is not None:
                self._publish_current_chart(sftp, current_chart, f'{self._remote_path}data/')
        except Exception:
            logger.exception(
                'Uploading output files to %s failed. Check the SSH key (%s), '
                'the remote path (%s), and host reachability. If the error is an '
                'unknown host key, add it with: ssh-keyscan %s >> %s . '
                'Files will be re-uploaded next cycle.',
                server.host, server.key_path, server.remote_path,
                server.host, Path.home() / '.buzz' / 'known_hosts',
            )
        finally:
            if sftp:
                sftp.close()
            if client:
                client.close()
