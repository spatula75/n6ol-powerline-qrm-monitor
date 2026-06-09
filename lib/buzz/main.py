import gc
import json
import os
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, time
from functools import wraps
from math import pow, log10, ceil
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

import jinja2
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import paramiko
import sounddevice as sd
from jinja2 import FileSystemLoader
from numba import njit
from numpy import abs, uint32, zeros, array, cumsum


class Buzz:
    """
    Hackish class for sampling audio data off the KX3 using the computer's sound card, running some time-series
    analysis on that data, grabbing some weather statistics, dumping the whole works to a CSV file, generating
    some graphs from the CSV, and uploading it all to my own web server for convenient access/sharing.

    Note this class is VERY hacked-together and should not be taken as an example of how to perform any serious
    analysis.  I wrote this because I needed something to help me keep tabs on a utility pole that produces a great
    deal of QRM on HF, in an effort to look for any cycles or patterns of behavior that might have proven useful
    in determining exactly which utility pole was causing the trouble.

    I present it here merely in case it's useful to someone or has any ideas anyone might find useful.  I certainly
    don't guarantee any particular performance or make any claims about this being the right way or best way to do it.

    It's worth noting that an earlier version of this attempted to use Fast Fourier Transforms looking for a 120Hz
    signal and its harmonics; however, the pulses last a mere 1.5-3ms each with a very small amount of sound in
    between pulses, and it turns out that this does not lend itself well to analysis with FFT.  Instead, I ended up
    poking around until I had developed an algorithm wherein I multiply sample data by a series of coefficients then
    sum the multiplied data.  This, in turn, ends up looking a lot like the convolution algorithm, and might even BE
    the convolution algorithm, but I did not try making any optimizations or standardizations beyond this point,
    because (1) what I have gets the job done, and (2) PG&E finally showed up to take a look at their goddamned pole.
    """

    """
    requirements.txt:
    
    sounddevice~=0.4.6
    numpy~=1.26.4
    numba~=0.59.1
    tzdata
    matplotlib~=3.8.4
    paramiko~=3.4.0
    
    """
    input_device = sd.query_devices('Line In (Realtek(R) Audio), Windows DirectSound', 'input')
    input_device_index = input_device['index']
    sample_rate = 16000
    duration = 3
    measurements_to_take = 3
    audio_rf_conversion_db = -32  # approximate difference in amplitude between audio and RF levels
    distance_attenuation = 29.54  # approximate dB drop over the distance to the noise source
    noise_min_snr = 12
    noise_floor = -98
    noise_threshold = noise_floor + noise_min_snr

    weather_url = 'http://192.168.1.160:8998/api/tags/process.json?temp&hum&SolarRad&wspeed&wgust&avgbearing'

    last_day = 0
    path = 'C:\\Users\\passp'

    def _fetch_weather_data(self):
        # Retrieve weather data from Cumulus MX server running on the LAN
        with urllib.request.urlopen(self.weather_url) as response:
            data = json.loads(response.read())
            return data['temp'], data['hum'], data['SolarRad'], data['wspeed'], data['wgust'], data['avgbearing']

    def _smooth(self, data: list, points: int):
        # Calculate the moving average across data, for a given number of points
        ret = cumsum(data, dtype=float)
        ret[points:] = ret[points:] - ret[:-points]
        return ret[points - 1:] / points

    def read_date_csv_to_time_dict(self, input_filename: str) -> dict:
        time_to_snr = defaultdict(lambda: 0)
        with open(input_filename, 'r') as file:
            while line := file.readline():
                timestamp, snr, signal, noise, temp, humidity, radiation = line.split(',')
                timestamp = datetime.fromisoformat(timestamp).astimezone(ZoneInfo('America/Los_Angeles'))
                time = timestamp.time()
                time = time.replace(minute=int(15 * (time.minute // 15)), second=0, microsecond=0)
                snr = float(snr)
                signal = float(signal)
                if signal >= self.noise_threshold and snr >= 15:
                    time_to_snr[time] += log10(snr) / log10(15)
        return {k: int(v) for k, v in time_to_snr.items()}

    @staticmethod
    def force_post_gc(func):
        # Workaround for https://github.com/matplotlib/matplotlib/issues/27713
        # The code in this class actually worked for about 2 years without ever hitting this bug, but
        # when it does hit, it's fatal.
        # (I suspect somewhat imperfectly, because you could probably still get a race between plt.close()
        # and this wrapper calling gc.collect())
        @wraps(func)
        def gc_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            gc.collect()

            return result

        return gc_wrapper

    @force_post_gc
    def generate_summary_graph_from_csv_range(self, output_filename: str, start_date: datetime):
        now_date = start_date
        zone = ZoneInfo('America/Los_Angeles')
        end_date = datetime.now(zone)

        time_to_snr = defaultdict(lambda: 0)
        while now_date <= end_date:
            now_date_str = now_date.strftime('%Y-%m-%d')
            now_date += timedelta(days=1)
            csv_filename = f'{self.path}/noise_data.{now_date_str}.csv'

            try:
                this_csv_time_to_snr = self.read_date_csv_to_time_dict(csv_filename)
                for (time, val) in this_csv_time_to_snr.items():
                    time_to_snr [time] += val
            except FileNotFoundError:
                pass

        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(1600 * px, 540 * px))

        plt.rcParams['timezone'] = 'America/Los_Angeles'
        run_time = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=zone)

        all_datetimes = []
        dt = run_time
        for x in range(4 * 24):
            all_datetimes.append(dt)
            dt = dt + timedelta(minutes=15)
        vals = [time_to_snr.get(dt.time(), 0) for dt in all_datetimes]

        # Convert values to percentages of the maximum value
        max_val = max(vals)
        if max_val == 0:
            return
        normalized_vals = [int(100 * (val / max_val)) for val in vals]

        # Sky blue is #87ceeb
        red_fade_range = 0xfe - 0x87
        green_fade_range = 0xfe - 0xce
        blue_fade_range = 0xfe - 0xeb

        colors = ['firebrick' if val == 100
                  else 'indianred' if val > 92
                  else 'lightcoral' if val > 85
                  else f'#{int(0x87 + (85 - val)/85 * red_fade_range):x}'
                       f'{int(0xce + (85 - val)/85 * green_fade_range):x}'
                       f'{int(0xeb + (85 - val)/85 * blue_fade_range):x}' for val in normalized_vals]

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.set_xlim(all_datetimes[0] - timedelta(minutes=10), all_datetimes[-1] + timedelta(minutes=10))

        ax.set_xlabel('Time (America/Los_Angeles zone)')
        ax.set_ylabel('Normalized Probability')

        ax.legend(title='Legend', handles=[
            mpatches.Patch(color='skyblue', label='<  85%'),
            mpatches.Patch(color='lightcoral', label='>  85%'),
            mpatches.Patch(color='indianred', label='>  92%'),
            mpatches.Patch(color='firebrick', label='= 100%')
        ])
        ax.bar(all_datetimes, normalized_vals, width=timedelta(minutes=13), color=colors)
        plt.title('Time of Day vs Normalized Probability of 120pps Interference\n'
                  f'15-minute increments from {start_date.strftime('%Y-%m-%d %H:%M')} '
                  f'to {end_date.strftime('%Y-%m-%d %H:%M')}')

        plt.tight_layout(pad=1.1)
        #plt.show()
        plt.savefig(output_filename)
        plt.close()

    @force_post_gc
    def generate_graph_from_csv(self, input_filename: str, output_filename: str, smooth=0):
        # Given a file full of CSV data generated from sampling/calculation, produce a nice graph of the data.
        with open(input_filename, 'r') as file:
            timestamps = []
            signals = []
            noises = []
            snrs = []
            while line := file.readline():
                timestamp, snr, signal, noise, _ = line.split(',', 4)
                try:
                    timestamp = datetime.fromisoformat(timestamp).astimezone(ZoneInfo('America/Los_Angeles'))
                    timestamps.append(timestamp)
                    signals.append(float(signal))
                    noises.append(float(noise))
                    snrs.append(float(snr))
                except ValueError:
                    # We'll get ValueError on the CSV headings, just skip it.
                    continue

        if smooth:
            if len(timestamps) <= smooth:
                return  # we can't do anything yet
            signals = self._smooth(signals, smooth)
            noises = self._smooth(noises, smooth)
            timestamps = timestamps[smooth - 1:]
            title = f'Powerline Noise vs Noise Floor ({smooth} point moving avg), {timestamps[0].strftime("%Y-%m-%d")} (America/Los_Angeles Timezone)'
        else:
            title = f'Powerline Noise vs Noise Floor, {timestamps[0].strftime("%Y-%m-%d")} (America/Los_Angeles Timezone)'

        signals_adjusted = [val + self.distance_attenuation
                            if ((val > self.noise_threshold and snrs[index] > self.noise_min_snr)
                                or (val > self.noise_threshold + 0.5 * self.noise_min_snr)) else val
                            for index, val in enumerate(signals)]

        #signals_adjusted = [val + self.distance_attenuation if val > self.noise_threshold else val
        #                    for val in signals]

        plt.rcParams['timezone'] = 'America/Los_Angeles'
        plt.tight_layout()

        px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
        figure, axes = plt.subplots(figsize=(1600*px, 640*px))
        plt.title(title)
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        noise_twin = axes.twinx()

        min_y_db = min(min(signals), min(noises), min(signals_adjusted), -48 + self.audio_rf_conversion_db) * 1.33
        max_y_db = max(max(signals), max(noises), max(signals_adjusted), -48 + self.audio_rf_conversion_db) / 1.33

        plot_signal, = axes.plot(timestamps, signals, 'r-', label='120pps dBm')
        #plot_signal_at_source, = axes.plot(timestamps, signals_adjusted,
        #                                   color='lightcoral', linestyle='dotted',
        #                                   label='est 120pps dBm at 10m')
        plot_noise, = noise_twin.plot(timestamps, noises, 'g-', label='Noise Floor dBm')

        axes.set_xlim(timestamps[0], timestamps[-1])
        axes.set_ylim(min_y_db, max_y_db)
        noise_twin.set_ylim(min_y_db, max_y_db)

        axes.set_xlabel('Time')
        axes.set_ylabel('dBm')
        noise_twin.get_yaxis().set_ticks([])

        axes.yaxis.label.set_color(plot_signal.get_color())
        noise_twin.yaxis.label.set_color(plot_noise.get_color())

        tick_kwargs = dict(size=4, width=1.5)
        axes.tick_params(axis='y', colors=plot_signal.get_color(), **tick_kwargs)
        noise_twin.tick_params(axis='y', colors=plot_noise.get_color(), **tick_kwargs)

        plot_s9 = axes.axhline(y=-73, color='tan', linestyle='dashed',
                               label='S9 (-73dBm) signal strength')
        plot_threshold = axes.axhline(y=self.noise_threshold, color='gray', linestyle='dashed',
                                      label=f"{self.noise_threshold} dBm threshold")
        plot_normal_floor = axes.axhline(y=self.noise_floor, color='gray',
                                         label=f"{self.noise_floor} dBm typical noise floor")

        axes.legend(loc='lower left', handles=[plot_signal, plot_noise, plot_s9, plot_threshold,
                                               plot_normal_floor])

        plt.savefig(output_filename, bbox_inches='tight', pad_inches=20 * px, pil_kwargs={'optimize': True})
        plt.close()

    def generate_index(self, output_filename, collection_time: datetime, image_path: str):
        template_path = Path(__file__).resolve().parent
        environment = jinja2.Environment(loader=FileSystemLoader(f'{template_path}/../../templates/'))
        template = environment.get_template('index.html')
        collection_time_formatted = collection_time.strftime('%d %B %Y %H:%M:%S %Z (%z)')
        # We want to skip the refresh tag at 11:59 PM to break browsers out of a refresh loop if someone leaves
        # a browser open.
        no_refresh = collection_time.timetz() == time(23, 59, 0, 0,
                                                      tzinfo=collection_time.tzinfo)
        content = template.render(filename=image_path, update_datetime=collection_time_formatted, no_refresh=no_refresh)
        with open(output_filename, mode="w", encoding="utf-8") as message:
            message.write(content)

    def run_collection(self):
        now = datetime.now(ZoneInfo('America/Los_Angeles')).replace(second=0, microsecond=0)

        snr_total = 0
        signal_db_total = 0
        noise_db_total = 0

        # Perform this measurement multiple times and then calculate averages
        for i in range(self.measurements_to_take):
            snr, signal_db, noise_db = self.sample_data
            snr_total += snr
            signal_db_total += signal_db
            noise_db_total += noise_db
        snr_mean = round(snr_total / self.measurements_to_take, 2)
        signal_mean = round(signal_db_total / self.measurements_to_take, 2) + self.audio_rf_conversion_db
        noise_mean = round(noise_db_total / self.measurements_to_take, 2) + self.audio_rf_conversion_db
        temperature, humidity, solar_radiation, wind_speed, wind_gust, wind_bearing = self._fetch_weather_data()

        # Dump the data to a CSV file (append-only)
        now_date_str = now.strftime('%Y-%m-%d')
        csv_filename = f'{self.path}/noise_data.{now_date_str}.csv'
        write_headers = not os.path.exists(csv_filename)
        csv_str = f'{now.isoformat()},{snr_mean:.2f},{signal_mean:.2f},{noise_mean:.2f},{temperature},{humidity},{solar_radiation},{wind_speed},{wind_gust},{wind_bearing}'
        with open(csv_filename, 'a+') as file:
            if write_headers:
                file.write('ISO datetime,120pps SNR,120pps signal dB,Noise floor dB,Temperature (F),Humidity (%),Solar radiation (w/m^2),Wind speed (MPH),Wind gust (MPH),Wind bearing (deg)\n')
            file.write(f'{csv_str}\n')

        # Now generate some plots.  For the first one, use no smoothing.
        plot_filename = f'{self.path}/noise_plot.{now_date_str}.png'
        self.generate_graph_from_csv(csv_filename, plot_filename)

        # For the second one, do a 6-point moving average.
        smooth_plot_filename = f'{self.path}/noise_plot_movavg.{now_date_str}.png'
        self.generate_graph_from_csv(csv_filename, smooth_plot_filename, 6)

        upload_files = [csv_filename, plot_filename, smooth_plot_filename]

        # These are expensive to generate when there's a lot of data, so only do it once an hour.
        if now.time().minute == 0:

            zone = ZoneInfo('America/Los_Angeles')
            start_date = datetime.fromisoformat('2024-05-15T00:00:00-0700')
            summary_filename = f'{self.path}/_noise_probability_summary.png'
            self.generate_summary_graph_from_csv_range(summary_filename, start_date)

            start_date = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
            start_date -= timedelta(days=7)
            summary_filename_7d = f'{self.path}/_noise_probability_summary_7d.png'
            self.generate_summary_graph_from_csv_range(summary_filename_7d, start_date)

            start_date = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
            start_date -= timedelta(days=30)
            summary_filename_30d = f'{self.path}/_noise_probability_summary_30d.png'
            self.generate_summary_graph_from_csv_range(summary_filename_30d, start_date)

            upload_files.extend([summary_filename, summary_filename_7d, summary_filename_30d])

        # Now upload the whole mess - CSV and all plots - to the server
        self.scp_to_server(upload_files, prefix='data/')

        # Generate the index page and upload it too
        index_filename = f'{self.path}/index.html'
        image_path = 'data/' + Path(smooth_plot_filename).name
        self.generate_index(index_filename, now, image_path)
        self.scp_to_server([index_filename])

        print(csv_str)

    @property
    def sample_data(self):
        # Record some audio
        recording = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1,
                           blocking=True, dtype='int16', device=self.input_device_index)
        # The audio format comes in an array of the format [ [channel_1..channel_n], ... ] even though we're
        # recording only one channel, so for ease of processing, turn this into a simple array of sample values.
        #
        # Note that this is also applying abs() to each sample value.  What we're trying to do is analyze
        # variances in amplitude for two sets of sample data, and if you want to get at the amplitude of a sample,
        # you want its absolute value rather than its raw value.  Obviously if we were doing any other kind of
        # signal analysis, this would be destroying information and probably would b0rk that analysis.
        # TODO: see if this can be done with np.array.reshape()
        mono_amplitude_array = array([abs(channel[0]) for channel in recording])

        output = self._calculate_pps_fit_array(mono_amplitude_array, self.sample_rate)

        # Find where we have the best "fit" of 120pps -- that is, where the calculations produced the largest value
        peak_offset_index = output.argmax()
        # Also find where we have the worst fit; we can use this to find the noise floor more effectively.
        min_offset_index = output.argmin()

        # now backtrack to the beginning of the sample data such that we remain aligned with the repeating pulse pattern
        # First, figure out where this pulse train actually began, because where we found the strongest signal
        # may not be at the start of the pulse train.
        peak_sample_frequency = (self.sample_rate / 120)
        peak_repeat_count = int(peak_offset_index / peak_sample_frequency)
        min_repeat_count = int(min_offset_index / peak_sample_frequency)
        first_peak_in_sequence_index = int(peak_offset_index - (peak_repeat_count * peak_sample_frequency))
        first_noise_floor_index = int(min_offset_index - (min_repeat_count * peak_sample_frequency))

        # now rerun the analysis just for the peak index and the peak offset index, over the *entire* set of sample data
        # (We're no longer trying to find a "fit", but rather to now find the sum for the pulse train across the entire
        # data set.)
        latest_analysis_index = max(first_peak_in_sequence_index, first_noise_floor_index)
        analysis_size = int((len(mono_amplitude_array) - latest_analysis_index) // peak_sample_frequency)
        output = self._calculate_pps_fit_array(mono_amplitude_array, self.sample_rate, analysis_size,
                                               first_peak_in_sequence_index, 1)
        avg_peak = output[0]  # we used a slide_len of 1, because we analyzed the entire sample set, so the output is just one element

        # Do the same thing for the "midpoint" data, where we suspect the values will be primarily noise, not pulse
        # train values.  In practice this does not seem to completely isolate the noise from the signal, possibly
        # because either I'm just doing something wrong here, because the pulse train is not the *only* noise being
        # thrown by the faulty equipment, or because receiving such a large magnitude pulse causes the receiver itself
        # to get a little noisy even in between pulses.
        output = self._calculate_pps_fit_array(mono_amplitude_array, self.sample_rate, analysis_size,
                                               first_noise_floor_index, 1)
        avg_noise = output[0]

        # now convert raw amplitude values into decibels
        db_reference = 20 * log10(pow(2, 16)/2)  # max value it can ever be
        db_peak = 20 * log10(avg_peak) if avg_peak > 0 else -128
        db_pulse_normalized = db_peak - db_reference
        db_background = 20 * log10(avg_noise) if avg_noise > 0 else -128
        db_background_normalized = db_background - db_reference
        snr = db_pulse_normalized - db_background_normalized

        return round(snr, 2), db_pulse_normalized, db_background_normalized

    @staticmethod
    @njit
    def _calculate_pps_fit_array(mono_amplitude_array, sample_rate, analysis_size=60, start_index=0, slide_len=0):
        """
        Given an array of sample data and the sample rate at which it was made, iterate over the array producing
        a value at each point indicating the intensity of a 120-pulse-per-second train that begins at that point.

        Analysis creates a coefficient array which always starts on a pulse train.

        :arg analysis_size: the number of pulses to try to fit, default 60 (half a second at 120 pps)
        """
        peak_sample_frequency = sample_rate / 120

        coefficient_len = ceil(analysis_size * peak_sample_frequency)
        coefficients = zeros(coefficient_len, dtype=uint32)
        # Set the pulses in the coefficient array.
        # Trying three 1's in a row to best fit the shape of the pulse and avoid mis-detecting other transients
        for i in range(0, analysis_size):
            pos = int(i * peak_sample_frequency)
            coefficients[pos] = 1
            coefficients[pos + 1] = 1
            coefficients[pos + 2] = 1
        # Walk the entire set of sample data looking for the best overall match of the coefficient window
        # If unset (0), default slide_len to the length of the entire set of sample data.
        if slide_len == 0:
            slide_len = len(mono_amplitude_array) - coefficient_len + 1

        # Note that this is starting to look an awful lot like a convolution algorithm, minus the 'flip' aka correlation
        output = np.correlate(mono_amplitude_array[start_index:], coefficients, mode='valid')[0:slide_len]

        # convert back to average amplitude by dividing by the number of 1 pulse coefficients
        output = output // (3 * analysis_size)
        return output

    def scp_to_server(self, files: list[str], prefix=''):
        sftp = None
        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            key_path = Path(__file__).parent / "buzz.pem"

            client.connect('192.168.1.123', username='spatula', password='', key_filename=str(key_path))

            sftp = client.open_sftp()
            for file in files:
                destination_name = Path(file).name
                sftp.put(file, f'/web/n6ol/noise/{prefix}{destination_name}')
        except BaseException as e:
            print(f'Got {e} when trying to copy files')
        finally:
            if sftp:
                sftp.close()
            if client:
                client.close()

    def collection_loop(self):
        while True:
            try:
                now = datetime.now(ZoneInfo('America/Los_Angeles'))
                next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                while now.timestamp() < next_minute.timestamp():
                    wait_seconds = next_minute.timestamp() - now.timestamp()
                    sleep(wait_seconds)
                    now = datetime.now(ZoneInfo('America/Los_Angeles'))

                self.run_collection()
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(f'Unexpected exception {e}, ignoring to try again next time.')


if __name__ == '__main__':
    Buzz().collection_loop()
