from dataclasses import dataclass
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent


@dataclass
class BuzzConfig:
    # Sounddevice name of the audio input recording the RF-to-audio converted signal.
    input_device_name: str = 'Line In (Realtek(R) Audio), Windows DirectSound'
    # Audio sample rate in Hz. Must match what the input device is configured to use.
    sample_rate: int = 16000
    # Length of each audio recording in seconds. Longer = more pulses to average over.
    duration: int = 3
    # Number of recordings averaged together per CSV entry. Higher = less noisy readings.
    measurements_to_take: int = 3
    # dB offset applied to audio amplitude to approximate RF level at the receiver input.
    # Hardware-specific: derived by calibrating against a known signal level.
    audio_rf_conversion_db: float = -32.0
    # Path loss in dB from the powerline to the monitoring location.
    # Used to estimate source strength from the measured level.
    distance_attenuation: float = 29.54
    # Minimum SNR in dB to count a sample as interference-present in the summary graphs.
    noise_min_snr: float = 12.0
    # Receiver noise floor in dBm. Combined with noise_min_snr to set noise_threshold.
    noise_floor: float = -98.0
    # IANA timezone name for CSV timestamps and graph labels.
    timezone: str = 'America/Los_Angeles'
    # Local directory where CSV files, plots, and the index page are written.
    path: str = r'C:\Users\passp'
    # CumulusMX JSON endpoint. Tags in the query string select which fields are returned.
    weather_url: str = 'http://192.168.1.160:8998/api/tags/process.json?temp&hum&SolarRad&wspeed&wgust&avgbearing'
    # Hostname or IP of the web server that hosts the published output.
    server_host: str = '192.168.1.123'
    server_username: str = 'spatula'                        # SSH username on the web server
    server_remote_path: str = '/web/n6ol/noise/'            # Remote path for uploaded data files
    server_key_path: str = str(_MODULE_DIR / 'buzz.pem')    # SSH private key for SCP authentication
    # ISO 8601 start date for the all-time summary graph.
    summary_start_date_iso: str = '2024-05-15T00:00:00-0700'
    # Powerline interference pulse rate in Hz: 120 for 60 Hz grid (North America),
    # 100 for 50 Hz grid (Europe and most of the rest of the world).
    pulse_rate: int = 120

    @property
    def noise_threshold(self) -> float:
        return self.noise_floor + self.noise_min_snr
