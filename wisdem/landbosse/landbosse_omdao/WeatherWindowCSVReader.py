from math import ceil

import pandas as pd

SEASON_WINTER = "winter"
SEASON_SPRING = "spring"
SEASON_SUMMER = "summer"
SEASON_FALL = "fall"


month_numbers_to_seasons = {
    1: SEASON_WINTER,
    2: SEASON_WINTER,
    3: SEASON_WINTER,
    4: SEASON_SPRING,
    5: SEASON_SPRING,
    6: SEASON_SPRING,
    7: SEASON_SUMMER,
    8: SEASON_SUMMER,
    9: SEASON_SUMMER,
    10: SEASON_FALL,
    11: SEASON_FALL,
    12: SEASON_FALL,
}


def read_weather_window(weather_data, local_timezone="America/Denver"):
    """
    This function converts a wind toolkit (WTK) formatted dataframe into
    a dataframe suitable for calculations.

    The .csv should have the first 5 columns of:

    Date, Temp C, Pressure atm, Direction deg, Speed m per s

    Other columns are ignored, headers are ignored and the first
    four lines are skipped. Dates in this file are assumed to be
    UTC. All columns which contain numeric only data are cast to
    float.

    It parses the local version of the date into year, month, day,
    hour. It also labels hours between 8 and 18 inclusive as 'normal'
    and hours from 18 to 7 as 'long'.

    The columns returned in the dataframe are:

    'Date UTC': The date and time in UTC of the measurements in that row.

    'Date': The date, localized to the timezone specified in the
        local_timezone parameter. See parameter list below.

    'Year': An integer of the year of the local time zone date

    'Month': An integer of the month of the local time zone date

    'Day': An integer of the day of the local time zone date

    'Hour': An integer of the hour of the local time zone date

    'Time window': If the integer hour is between 8 and 18 inclusive,
        this is 'normal'. For hours outside of that range, this is
        'long'.

    'Season': Season of the year. Months 1, 2, 3 are winter; months 4, 5, 6
        are spring; months 7, 8, 9 are summer; months 10, 11, 12 are fall.

    'Pressure atm': Air pressure in atm.

    'Direction deg': Wind direction in degrees.

    'Speed m per s': Wind speed in meters per second.

    Parameters
    ----------
    filename : str
        The filename to read for the csv.

    local_timezone : str
        The local timezone. The is a TZ database name for the timezone.
        Find the TZ database listing at
        https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    Returns
    -------
    pd.DataFrame
        A pandas data frame made from the CSV
    """
    # set column names for weather data and keep only the renamed columns
    column_names = weather_data[4:].columns
    renamed_columns = {
        column_names[0]: "Date UTC",
        column_names[1]: "Temp C",
        column_names[2]: "Pressure atm",
        column_names[3]: "Direction deg",
        column_names[4]: "Speed m per s",
    }
    weather_data = weather_data[4:].rename(columns=renamed_columns)
    weather_data = weather_data.reset_index(drop=True)
    weather_data = weather_data[renamed_columns.values()]

    # Parse the datetime data and localize it to UTC
    weather_data["Date UTC"] = pd.to_datetime(weather_data["Date UTC"]).dt.tz_localize("UTC")

    # Convert UTC to local time
    weather_data["Date"] = weather_data["Date UTC"].dt.tz_convert(local_timezone)

    # Extract month, day, hour from the local date
    weather_data["Month"] = weather_data["Date"].dt.month
    weather_data["Day"] = weather_data["Date"].dt.day
    weather_data["Hour"] = weather_data["Date"].dt.hour

    # The original date columns are now redundant. Drop them.
    weather_data.drop(columns=["Date", "Date UTC"])

    # create time window for normal (8am to 6pm) versus long (24 hour) time window for operation
    weather_data["Time window"] = weather_data["Hour"].between(8, 18, inclusive="both")
    boolean_dictionary = {True: "normal", False: "long"}
    weather_data["Time window"] = weather_data["Time window"].map(boolean_dictionary)

    # Add a seasons column
    weather_data["Season"] = weather_data["Month"].map(month_numbers_to_seasons)

    # Cast the columns that are numeric to float64
    columns_to_cast = ["Pressure atm", "Direction deg", "Speed m per s"]
    for column_to_cast in columns_to_cast:
        weather_data[column_to_cast] = pd.to_numeric(weather_data[column_to_cast], downcast="float")

    # return the result
    return weather_data


def extend_weather_window(weather_window_df, months_of_weather_data_needed):
    """
    This function extends a weather window by duplicating the rows to create
    a weather window that spans the needed number of months.

    Suppose that a weather window is 8760 hours long, but that 72 months
    (52560 hours) are needed. This method will create a new dataframe
    that has the original 8760 hours duplicated 6 times.

    However, if the number of needed months is less than or equal to the
    duration of the weather window, a reference to the original weather
    window is returned. Also, the number of rows returned will always be
    a multiple of the number of rows in the original data frame.

    If rows are added to the weather window, they are added to a new dataframe.
    The weather window is not modified in place.

    Parameters
    ----------
    weather_window_df : pd.DataFrame
        The original weather window.

    months_of_weather_data_needed : int
        The number of months of weather data needed. Each month is approximated
        to have 730 hours.

    Returns
    -------
    pd.DataFrame
        If the weather window accommodates the necessary number of months, then
        a reference to the original dataframe is returned. Otherwise, a reference
        to a new dataframe that has a row count that is a multiple of the number
        of rows in the original weather window.
    """
    hours_per_month = 730
    hours_of_weather_data_needed = hours_per_month * months_of_weather_data_needed
    hours_of_weather_data_available = len(weather_window_df)

    if hours_of_weather_data_needed <= hours_of_weather_data_available:
        return weather_window_df

    number_of_windows_needed = int(ceil(hours_of_weather_data_needed / hours_of_weather_data_available))
    weather_window_as_list = weather_window_df.to_dict(orient="records")
    weather_window_repeated_as_list = weather_window_as_list * number_of_windows_needed
    result = pd.DataFrame(weather_window_repeated_as_list)

    return result
