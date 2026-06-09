from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from buzz.main import Buzz
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


if __name__ == '__main__':

    buzz = Buzz()
    buzz.generate_summary_graph_from_csv_range(None)
    exit(0)
    start_date = now_date = datetime.fromisoformat('2024-05-15T00:00:00-0700')
    zone = ZoneInfo('America/Los_Angeles')
    end_date = datetime.now(zone)

    time_to_snr = defaultdict(lambda: 0)
    while now_date <= end_date:
        now_date_str = now_date.strftime('%Y-%m-%d')
        now_date += timedelta(days=1)
        csv_filename = f'{buzz.path}/noise_data.{now_date_str}.csv'

        this_csv_time_to_snr = buzz.read_date_csv_to_time_dict(csv_filename)
        for (time, val) in this_csv_time_to_snr.items():
            time_to_snr[time] += val

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(1600*px, 540*px))

    plt.rcParams['timezone'] = 'America/Los_Angeles'
    run_time = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=zone)

    sorted_time_to_snr = {k: time_to_snr[k] for k in sorted(time_to_snr.keys())}

    all_datetimes = []
    dt = run_time
    for x in range(4 * 24):
        all_datetimes.append(dt)
        dt = dt + timedelta(minutes=15)
    vals = [time_to_snr.get(dt.time(), 0) for dt in all_datetimes]
    max_val = max(vals)
    normalized_vals = [int(100 * (val / max_val)) for val in vals]

    colors = ['indianred' if val > 90 else 'salmon' if val > 80 else 'skyblue' for val in normalized_vals]

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.set_xlim(all_datetimes[0] - timedelta(minutes=10), all_datetimes[-1] + timedelta(minutes=10))

    ax.set_xlabel('Time (America/Los_Angeles zone)')
    ax.set_ylabel('Normalized Probability')

    ax.legend(title='Legend', handles=[
        mpatches.Patch(color='skyblue', label='< 80%'),
        mpatches.Patch(color='salmon', label='> 80%'),
        mpatches.Patch(color='indianred', label='> 90%')
    ])
    ax.bar(all_datetimes, normalized_vals, width=timedelta(minutes=13), color=colors)
    plt.title('Time of Day vs Normalized Probability of 120pps Interference\n'
              f'15-minute increments from {start_date.strftime('%Y-%m-%d %H:%M')} '
              f'to {end_date.strftime('%Y-%m-%d %H:%M')}')

    plt.tight_layout(pad=1.1)
    plt.show()
