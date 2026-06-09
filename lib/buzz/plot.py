import paramiko
from pathlib import Path
from buzz.main import Buzz

if __name__ == '__main__':
    buzz = Buzz()

    dates = ['2024-04-21', '2024-04-22', '2024-04-23', '2024-04-27']

    for date in dates:
        csv_filename = f'C:\\Users\\passp\\noise_data.{date}.csv'
        plot_filename = f'C:\\Users\\passp\\noise_plot_movavg.{date}.png'
        buzz.generate_graph_from_csv(csv_filename, plot_filename, 6)
        buzz.scp_to_server([csv_filename, plot_filename])
