from obspy.clients.filesystem.tsindex import Client as TSindex_Client

from obspy import UTCDateTime as UTC
import obspy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import time


from sqlalchemy.pool import NullPool
import sqlalchemy as sa
from obspy.clients.filesystem.tsindex import TSIndexDatabaseHandler


class NullPoolTSIndexDatabaseHandler(TSIndexDatabaseHandler):
    def __init__(self, database, *args, **kwargs):
        super().__init__(database, *args, **kwargs)

        # Recreate engine with NullPool
        db_path = f"sqlite:///{self.database}"
        self.engine = sa.create_engine(db_path, poolclass=NullPool)
        self.session = sa.orm.sessionmaker(bind=self.engine)


def get_single_channel_availability(
    client, network, station, location, channel, max_chunk_days=365, sleep_time=0.0
):
    """
    Get full availability for a given channel across potentially many years.

    Parameters:
        client: TSindex_Client instance
        network: str
        station: str
        location: str
        channel: str
        max_chunk_days: int, default 365 — how many days to request per chunk
        sleep_time: float, default 0.0 — time to sleep between requests (in seconds)

    Returns:
        Pandas DataFrame with availability data
    """
    extent = client.get_availability_extent(
        network=network,
        station=station,
        location=location,
        channel=channel,
    )

    if not extent:
        print("No availability extent found.")
        return pd.DataFrame()
    if len(extent) > 1:
        print("Multiple extents found, must specify a single station/channel.")
        return pd.DataFrame()

    start_extent = extent[0][4]
    end_extent = extent[0][5]
    # print(f"Data extent: {start_extent.date} to {end_extent.date}")

    all_data = []

    current_start = start_extent  # UTC datetime
    while current_start < end_extent:
        end_range = current_start + max_chunk_days * 86400  # 86400 seconds in a day
        current_end = min(end_range, end_extent)
        # Queries always start at 00:00 and end at 23:59, regardless of input time
        # So, end time should round up to 23:59:59 UTC of the end date
        current_end = current_end.replace(hour=23, minute=59, second=59, microsecond=0)
        # print(f"Fetching from {current_start} to {current_end}...")
        try:
            chunk = client.get_availability(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=UTC(current_start),
                endtime=UTC(current_end),
                merge_overlap=True,
            )
            all_data.extend(chunk)
        except Exception as e:
            print(
                f"Failed to get {network}.{station}.{location}.{channel} availability from {current_start.date} to {current_end.date}. \n\
                    {e}"
            )
            # Extended error message
            # \t--if getting QueuePool limit, try setting larger max_chunk_days (100+) or adding a sleep_time \n\
            # \t--if 'float object cannot be interpreted as an integer' OR 'max recursion depth exceeded', try setting lower max_chunk_days \n\

        current_start = (
            current_end + 2
        )  # Add two seconds, so next query starts at 00:00:01 UTC of the next date
        time.sleep(
            sleep_time
        )  # Avoid overwhelming the server with requests, try 0.5 if exceeding QueuePool limit

    # Convert to DataFrame
    df = pd.DataFrame(
        all_data,
        columns=[
            "network",
            "station",
            "location",
            "channel",
            "start_time",
            "end_time",
        ],
    )
    df["start_time"] = df["start_time"].apply(lambda x: x.datetime)
    df["end_time"] = df["end_time"].apply(lambda x: x.datetime)
    return df


def compute_uptime_percentage(df, interval_days=1):
    uptime_rows = []

    # Get the minimum date as anchor for interval alignment
    global_start = pd.Timestamp(
        "1970-01-01"
    )  # Set epoch start, so intervals align for all channels

    for _, row in df.iterrows():
        current = row["start_time"]
        end = row["end_time"]

        while current < end:
            # Align current to nearest interval block starting from global_start
            delta_days = (
                current.normalize() - global_start
            ).days  # Total number of days between global_start and current (midnight)
            aligned_block = (
                delta_days // interval_days
            )  # Which interval block this 'current' timestamp falls into (0-based)
            period_start = global_start + pd.Timedelta(
                days=aligned_block * interval_days
            )  # Start of the aligned interval block
            period_end = period_start + pd.Timedelta(
                days=interval_days
            )  # End of the interval block (exclusive)

            # Clip to actual data
            segment_start = max(
                current, period_start
            )  # Start of the usable segment: whichever is later
            segment_end = min(
                end, period_end
            )  # End of the usable segment: whichever is earlier
            duration = (
                segment_end - segment_start
            ).total_seconds()  # Actual uptime duration within this interval block (in seconds)

            uptime_rows.append(
                {
                    "period": period_start,
                    "duration": duration,
                    "network": row["network"],
                    "station": row["station"],
                    "location": row["location"],
                    "channel": row["channel"],
                    "period_seconds": interval_days * 86400,
                }
            )

            current = period_end

    # Create and group DataFrame
    uptime_df = pd.DataFrame(uptime_rows)

    grouped = (
        uptime_df.groupby(["period"])
        .agg({"duration": "sum", "period_seconds": "first"})
        .reset_index()
    )
    grouped["uptime_percent"] = grouped["duration"] / grouped["period_seconds"]
    return grouped[["period", "uptime_percent"]]


def plot_uptime(df, interval_days=1):
    """
    Plots horizontal bars for each period, with the color based on the uptime percentage.
    Each station will be represented along the y-axis, and the x-axis will represent the date.

    Parameters:
    df (DataFrame): The dataframe containing uptime percentages for various stations
    """
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize for coloring
    colors = [
        (255 / 255, 165 / 255, 0 / 255),  # Orange
        (255 / 255, 230 / 255, 100 / 255),  # Yellow
        (50 / 255, 205 / 255, 50 / 255),  # Lime Green
    ]
    cmap_name = "orange_to_green"
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
    norm = mcolors.Normalize(vmin=0, vmax=100)  # Uptime percentage ranges from 0 to 100

    # Create a list to hold the station names
    stations = list(df.columns[1:])  # Exclude the 'period' column
    num_stations = len(stations)

    # Create a list of periods (dates)
    periods = df["period"]

    # Loop through each station and plot horizontal bars
    for i, station in enumerate(stations):
        # Get the uptime values for this station
        uptime_values = df[station]

        # Plot each period as a horizontal bar
        for j, (period, uptime) in enumerate(zip(periods, uptime_values)):
            if not pd.isna(uptime):  # Ignore NaN values
                ax.barh(
                    i,  # Station index (y position)
                    interval_days,  # the width is in days
                    left=period,  # Set the left side of the bar to the starting point of the period
                    color=cmap(norm(uptime * 100)),  # Color based on uptime percentage
                    height=0.8,  # Height of each bar
                    label=(
                        station if j == 0 else ""
                    ),  # Only label the first bar for each station
                )

    # Set the y-axis labels to station names
    ax.set_yticks(np.arange(num_stations))
    ax.set_yticklabels(stations)
    ax.set_ylim(-0.6, num_stations - 0.4)  # Adjust y-limits to fit the bars

    # Set labels and
    ax.set_xlabel("Date")
    ax.set_ylabel("Stations")
    ax.set_title(f"Data Availability by Station")

    # Format the x-axis dates
    ax.set_xlim(
        periods.min() - pd.Timedelta(days=interval_days),
        periods.max() + 2 * pd.Timedelta(days=interval_days),
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Place major ticks every year
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Format ticks as years
    ax.xaxis.set_minor_locator(
        mdates.MonthLocator()
    )  # Place minor ticks for each month

    # Create a colorbar based on the colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"Availability (%, {interval_days}-day intervals)")

    plt.tight_layout()

    return fig, ax


def availability_plot(
    tsindex_path,
    network,
    station,
    location,
    channel,
    interval_days=1,
    max_chunk_days=100,
    queue_pool_sleep_time=0.0,
):
    """
    Plot availability for all channels matching the given parameters.

    Parameters:
        tsindex_path: Path to a timeseries.sqlite file, used to create TSindex_Client instance
        network: str
        station: str
        location: str
        channel: str
        interval_days: int, default 1 — how many days to group together to calculate availability %
        max_chunk_days: int, default 365 — how many days to request per chunk
    """
    try:
        # Pass in the
        tsindex = NullPoolTSIndexDatabaseHandler(database=tsindex_path)
        client = TSindex_Client(
            tsindex
        )  # Initialize the client with the modified nullpool handler
    except Exception as e:
        print(
            f"Unable to create TSindex_Client instance. Check that a valid .sqlite file exists at {tsindex_path}"
        )
        print(e)
        return

    inventory = (
        client.get_nslc(  # Gets the inventory in the sqlite file matching query params
            network=network, station=station, location=location, channel=channel
        )
    )
    # print(inventory)
    if not inventory:
        print("No inventory found.")
        return
    else:
        print(f"Found {len(inventory)} channels matching query.")

    channel_uptimes = pd.DataFrame()  # merge into channel_uptimes
    for channel in inventory:
        channel_name = ".".join(channel)
        print(f"Fetching availability  --  {channel_name}")

        df_avail = get_single_channel_availability(
            client,
            network=channel[0],
            station=channel[1],
            location=channel[2],
            channel=channel[3],
            max_chunk_days=max_chunk_days,
            sleep_time=queue_pool_sleep_time,
        )
        if df_avail.empty:
            print(f"No availability data for {channel}.")
            continue

        df_uptime = compute_uptime_percentage(df_avail, interval_days=interval_days)
        # Rename the uptime_percent column to include the channel name
        df_uptime = df_uptime.rename(columns={"uptime_percent": channel_name})

        # merge togehter into channel_uptimes
        if channel_uptimes.empty:
            # If this is the first DataFrame, initialize channel_uptimes with the current DataFrame
            channel_uptimes = df_uptime
        else:
            # Merge on the 'period' column (which represents the start of each period)
            channel_uptimes = pd.merge(
                channel_uptimes,
                df_uptime[
                    ["period", channel_name]
                ],  # Keep only 'period' and the current channel's uptime
                on="period",  # Merge on the 'period' column
                how="outer",  # Use outer join to keep all periods from all DataFrames
            )
    channel_uptimes = channel_uptimes.sort_values("period")

    fig, ax = plot_uptime(channel_uptimes, interval_days=interval_days)

    return fig, ax

    # Now we are ready to plot


if __name__ == "__main__":
    tsindex = (
        r"C:\Users\csaunders-shultz\Documents\data\rover_database\timeseries.sqlite"
    )

    # start_time = time.time()

    fig, ax = availability_plot(
        tsindex,
        network="",
        station="",
        location="",
        channel="",
        interval_days=1,
        max_chunk_days=200,
        queue_pool_sleep_time=0.0,
    )

    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.4f} seconds")

    plt.show()
    plt.close()

"""
Testing notes:
The NullPoolTSIndexDatabaseHandler class is a subclass of TSIndexDatabaseHandler that uses NullPool instead of QueuePool.

Times to process my entire SQlite database (18 channels) with interval_days=1, sleep_time=0.0
max_chunk_days      QueuePool (original)      NullPool (modified)
100                 14.4 seconds             15.0 seconds
50                  15.6 seconds             14.8 seconds
200                 13.5 seconds             13.3 seconds
400                 14.0 seconds             14.2 seconds
800                 14.0 seconds             14.9 [THR Error]            
25                  [QPL]                    34.8 seconds
1600                 14.5 seconds            14.4 seconds

Error codes:
QPL     - QueuePool limit of size 5 overflow 10 reached, connection timed out, timeout 30.00
        (results in missing data in the plot, and very slow runtime)

THR     sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread.
        (plot still looks okay after this error)


"""
print("")
