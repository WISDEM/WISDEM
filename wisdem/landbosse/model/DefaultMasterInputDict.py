class DefaultMasterInputDict:
    """
    DefaultMasterInput is a class that handles all the default values
    for the master input dictionary.
    """

    def __init__(self):
        """
        This constructor creates the defaults master input dictionary.
        This is called default_input
        """
        self.default_input_dict = dict()

        # Set the list and dictionary values on the master dictionary
        self.default_input_dict["season_construct"] = ["spring", "summer", "fall"]
        self.default_input_dict["time_construct"] = "normal"
        self.default_input_dict["hour_day"] = {"long": 24, "normal": 10}
        self.default_input_dict["operational_construction_time"] = self.default_input_dict["hour_day"][
            self.default_input_dict["time_construct"]
        ]

    def populate_input_dict(self, incomplete_input_dict):
        """
        Completely fills the input_dict. If there are any keys in the
        default input dictionary that are not on incomplete_input_dict, those
        missing key/value pairs are placed on a fully populated input
        dictionary.

        This method uses the self.__default_input_dict for the values
        it fills in the output dictionary.

        Parameters
        ----------
        incomplete_input_dict : dict
            The dict of input values that may or may not be complete.

        Returns
        -------
        dict
            The fully populated output diction
        """
        complete_dict = dict()
        for key, value in incomplete_input_dict.items():
            complete_dict[key] = value
        for key, value in self.default_input_dict.items():
            if key not in complete_dict:
                complete_dict[key] = value
        return complete_dict
