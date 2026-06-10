from datetime import datetime, time
from pathlib import Path

import jinja2
import paramiko
from jinja2 import FileSystemLoader

from buzz.config import BuzzConfig

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / 'templates'


class Publisher:
    def __init__(self, config: BuzzConfig):
        self._config = config
        environment = jinja2.Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)))
        self._template = environment.get_template('index.html')

    def generate_index(self, output_filename: str, collection_time: datetime, image_path: str):
        collection_time_formatted = collection_time.strftime('%d %B %Y %H:%M:%S %Z (%z)')
        no_refresh = collection_time.timetz() == time(23, 59, 0, 0, tzinfo=collection_time.tzinfo)
        content = self._template.render(
            filename=image_path,
            update_datetime=collection_time_formatted,
            no_refresh=no_refresh,
        )
        with open(output_filename, mode='w', encoding='utf-8') as f:
            f.write(content)

    def scp_to_server(self, files: list[str], prefix=''):
        sftp = None
        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self._config.server_host, username=self._config.server_username,
                           password='', key_filename=self._config.server_key_path)
            sftp = client.open_sftp()
            for file in files:
                destination_name = Path(file).name
                sftp.put(file, f'{self._config.server_remote_path}{prefix}{destination_name}')
        except BaseException as e:
            print(f'Got {e} when trying to copy files')
        finally:
            if sftp:
                sftp.close()
            if client:
                client.close()
