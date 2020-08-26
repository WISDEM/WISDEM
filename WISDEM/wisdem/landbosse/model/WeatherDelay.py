import numpy as np
import pandas as pd


class WeatherDelay:
    """
    Calculates weather delays for a project based on weather data, season of
    construction, operational hours, mission time,
    and critical wind speed -- only calculates wind delays right now.

    Central to this calculation is a weather window, which is a list of hours
    during the times when construction can happen. Each row lists
    an year, month, day and hour with the various weather conditions. Below,
    you can specify a start delay, which is the number of rows in the data--
    or the number of hours at the start of the weather window--during which the
    work will not start.

    The INPUT keys are the following:

    weather_window
        (pd.DataFrame) The weather window as prepared by the
        read_weather_window function in the WeatherWindowCSVReader module.
        See the documentation for that function for details on the
        columns of the weather window.

    start_delay_hours
        (float) Delay of mission from start of weather window. The weather
        window is a list of hours with the weather values as rows. The delay
        can be as little as 0 hours or the total available construction hours
        for the project.

    mission_time_hours
        (float) Length of mission. Mission length is the time it takes to
        complete the operation that may be delayed by the weather.

    critical_wind_speed_m_per_s
        (float) Wind speed that the mission must shutdown and enter a delay state.


    wind_height_of_interest_m
        (float) Height used in wind shear calculations.

    The OUTPUT keys are the following

    wind_delay
        (list) List of number of hours of each wind delay during the mission.
        length of list is number of weather delays. Value in list is duration
        of weather delay in hours.

    Parmeters
    ---------
    input_dict : dict
        The input data

    output_dict : dict
        The output data
    """

    def __init__(self, input_dict, output_dict):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.validate_inputs(self.input_dict)
        self.output_dict['wind_delays'] = self.calculate_wind_delay()

    def validate_inputs(self, input_dict):
        """
        This method checks a dictionary to make sure it has keys for all
        necessary values needed for calculations in an instance of this
        class. It is made to validate input_dict dictionaries.

        Returns
        -------
        None
            This method returns nothing if the validation passes. If it does
            not pass an exception is raised.

        Raises
        ------
        ValueError
            If one of the keys is missing, this method raises a ValueError
        """
        required_keys = {
            'start_delay_hours',
            'mission_time_hours',
            'critical_wind_speed_m_per_s',
            'wind_height_of_interest_m',
            'wind_shear_exponent',
            'weather_window'
        }
        found_keys = set(input_dict.keys())
        if len(required_keys - found_keys) > 0:
            err_msg = '{}: did not find all required keys in inputs dictionary. Missing keys are {}'
            raise ValueError(err_msg.format(type(self).__name__, required_keys - found_keys))

    def calculate_wind_delay(self):
        """
        Calculates wind delay based on weather window, mission time, and critical wind speed.

        This method must be executed after the create_weather_window method which calculates
        the weather window. This method assumes that self.output['weather_window'] is set.
        That is where it gets its weather window.

        Calculate wind delay based on weather window, operation start
        delay, mission time, and critical wind speed

        Returns
        -------
        list
            Number of hours for each wind delay encountered during mission.
            count of list = number of weather delays.
            values in list = durations of weather delays.
        """

        # Pull function variables off of the dictionaries to make the
        # following lines shorter. Also, the keys on the input
        # dictionary have units but the lines below do not have units in
        # the variable names.
        start_delay = self.input_dict['start_delay_hours']
        mission_time = self.input_dict['mission_time_hours']
        critical_wind_speed = self.input_dict['critical_wind_speed_m_per_s']
        wind_height_of_interest_m = self.input_dict['wind_height_of_interest_m']
        wind_shear_exponent = self.input_dict['wind_shear_exponent']
        weather_window = self.input_dict['weather_window']

        # Extract only the 'Speed m per s' as a dataframe, and only retain
        # elements where index is > start_delay and < mission_time
        wind_speeds_m_s = weather_window['Speed m per s'].values
        # check if mission time exceeds size of weather window
        if mission_time > len(wind_speeds_m_s):
            raise ValueError('{}: Error: Mission time longer than weather window'.format(type(self).__name__))
        wind_speeds_m_s_filtered = wind_speeds_m_s[(start_delay + 1):(int(mission_time) + 1)]

        # Calculate the wind speed at the particular, given the wind shear exponent
        wind_speed_at_height_m_s = wind_speeds_m_s_filtered * (wind_height_of_interest_m / 100) ** wind_shear_exponent

        # wind_delays is an array of booleans. It is True if the critical
        # wind speed is exceeded. False if the critical wind speed is not
        # exceeded. Each element represents an hour of wind
        wind_delays = wind_speed_at_height_m_s > critical_wind_speed

        # If there are any wind delays found:
        #
        # Iterate over each of the wind_delays. Count for contiguous blocks of
        # hours of wind delays. Add the length of each of these blocks to the
        # delay_durations.
        #
        # This code snippet takes O(n) linear time. The trick in vectorizing it
        # is that contiguous blocks of unknown length of True wind delays
        # are needed to find durations. But linear time may not be a problem
        # because the data we are iterating over has already been filtered down
        # to a small set.

        if np.any(wind_delays):

            # Holds the list of delay durations
            delay_durations = []

            # As we go through each delay, this accumulates the duration of
            # the current delay. Each True element of wind_delays increments
            # this by one
            current_delay_duration = 0

            # The following variable is True if we are iterating through
            # that are True, which means a weather delay.
            iterating_through_wind_delay = False

            # Iterate over each element of wind_delays. This is O(n) as
            # noted above.
            for wind_delay in np.nditer(wind_delays):
                # If we are in a weather delay...
                if wind_delay:
                    # If we are not currently iterating over a weather delay
                    # a new continuous sequence of delayed hours has started.
                    if not iterating_through_wind_delay:
                        current_delay_duration = 1
                        iterating_through_wind_delay = True

                    # While iterating over delay, increment counter.
                    else:
                        current_delay_duration += 1

                # If we are NOT iterating through a wind delay
                # And we were iterating through a wind delay, end that
                # delay and record the duration.
                elif iterating_through_wind_delay:
                    delay_durations.append(current_delay_duration)
                    iterating_through_wind_delay = False

                # Otherwise we are not iterating through a wind delay and
                # and did not finish a wind delay, so we do nothing with
                # no need for an else.

            # Finally return the durations we found.
            return delay_durations

        # If there are not wind delays, return a list with just 0 in it
        else:
            return [0]

    def run_module(self):
        """
        This method runs all other methods in the module in order to set the
        correct output keys.

        Returns
        -------
        int
            0 if the module ran without errors. 1 if there was an error.
        """
        try:
            self.output_dict['wind_delay'] = self.calculate_wind_delay()
            return 0    # module ran successfully
        except:
            return 1    # module did not run successfully

