import os
import re
from typing import Any, Dict, List, Union

import dearpygui.dearpygui as dpg

import wisdem.inputs.validation as val
from wisdem.glue_code.runWISDEM import run_wisdem


class DPGLineEdit(object):
    """
    Introduces a value edit on a line for the yaml dictionary leafs.  Updates referenced dictionary on callback.

    """

    def __init__(self, dictionary: Union[List[Any], Dict[str, Any]], key: str, value: Any, parent_tag: str) -> None:
        """
        The instance attributes for this class are:

        _mydict: Dict[str, Any]
            The dictionary to be changed when this text field is changed.

        _mykey: str
            The key on the dictionary to be changed when this text field
            changes.

        _list_re: re.Pattern
            The regex that matches a string that contains a list, so that
            it does not need to be instantiated each time it is used.

        Parameters
        ----------
        dictionary: Dict[str, Any]
            The dictionary to be changed.

        key: str
            The key whose value is to be changed on the dictionary.
        """
        self._mydict = dictionary
        self._mykey = key
        self._list_re = re.compile(r"^\[.*\]$")

        if self.is_list(str(value)):
            dpg.add_input_text(label=key, default_value=str(list(value)), callback=self.mycallback, parent=parent_tag)

        elif self.is_float(value):
            dpg.add_input_float(
                label=key, default_value=float(value), width=200, callback=self.mycallback, parent=parent_tag
            )

        elif self.is_integer(value):
            dpg.add_input_int(
                label=key, default_value=int(value), width=200, callback=self.mycallback, parent=parent_tag
            )

        elif self.is_boolean(value):
            dpg.add_input_text(
                label=key, default_value=str(bool(value)), width=150, callback=self.mycallback, parent=parent_tag
            )

        else:
            dpg.add_input_text(label=key, default_value=str(value), callback=self.mycallback, parent=parent_tag)

    def mycallback(self, sender, app_data) -> None:
        """
        Callback on line item entry to update referenced dictionary
        """
        if self._mydict is not None and self._mykey is not None:
            if self.is_list(str(app_data)):
                value = self.parse_list(app_data)

            elif self.is_float(app_data):
                value = float(app_data)

            elif self.is_integer(app_data):
                value = int(app_data)

            elif self.is_boolean(app_data):
                value = self.parse_boolean(app_data)

            else:
                value = str(app_data)

            self._mydict[self._mykey] = value

    def parse_list(self) -> List[Union[str, float]]:
        """
        This parses the text in the field to a list of numbers and/or strings.

        Returns
        -------
        List[Union[str, float]]
            A list that contains strings and floats, depending on what could be
            parsed out of the text in the field.
        """
        trimmed_text = self.text()[1:-1]
        trimmed_str_values = [x.strip() for x in trimmed_text.split(",")]
        result: List[Union[str, float]] = []
        for x in trimmed_str_values:
            if self.is_integer(x):
                result.append(int(x))
            elif self.is_float(x):
                result.append(float(x))
            else:
                result.append(x)
        return result

    @staticmethod
    def parse_boolean(value: str) -> bool:
        """
        This method parses a string as a boolean. See the is_boolean()
        method below.

        Parameters
        ----------
        value: str
            The value that is being parsed as a boolean.

        Returns
        -------
        bool
            The value, parsed as a boolean.
        """

        return True if str(value).lower() == "true" else False

    @staticmethod
    def is_float(value: Any) -> bool:
        """
        This tests if a value is a float and returns True or False depending
        on the outcome of the test

        Parameters
        ----------
        value: Any
            The value to test.

        Returns
        -------
        bool
            True if the value is a float, False otherwise.
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_integer(value: Any) -> bool:
        """
        This tests if a value is an integer and returns True or False depending
        on the outcome of the test

        Parameters
        ----------
        value: Any
            The value to test.

        Returns
        -------
        bool
            True if the value is an integer, False otherwise.
        """
        try:
            int(value)
            return True
        except ValueError:
            return False

    def is_list(self, value: str) -> bool:
        """
        Determines whether a value is a list through a regular expression
        match. The intent is that, if the value is a list, it will be
        parsed as a list.

        Parameters
        ----------
        value: str

        Returns
        -------
        bool
            True if the value appears to be a list, false otherwise
        """
        return self._list_re.match(value) is not None

    def is_boolean(self, value: str) -> bool:
        """
        Determines whether a string encodes a "True" or a "False" value. It
        does not parse the boolean value. If this method returns True, then
        the value can be parsed as a True or False boolean.

        Both Pythonic True and False and JSON-style true and false are
        accepted as booleans.

        Note: This should probably be checkbox, not a text box, but this
        is a temporary workaround.

        Parameters
        ----------
        value: str
            The string to be tested.

        Returns
        -------
        bool
            True if the string represents a boolean, false otherwise.
        """
        return str(value).lower() in ["true", "false"]


class GUI_Master(object):
    def __init__(self):
        self._input_mode = True

        self.geom_dict = {}
        self.model_dict = {}
        self.anal_dict = {}
        self.output_dict = {}
        self.froot_export = "gui_export"
        self.working_dir = os.path.expanduser("~")
        self.tag_list = []

        # Load in starting point yaml files
        temp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
            "examples",
            "02_reference_turbines",
        )
        temp_geom = os.path.join(temp_dir, "IEA-15-240-RWT.yaml")
        temp_modopt = os.path.join(temp_dir, "modeling_options.yaml")
        temp_analopt = os.path.join(temp_dir, "analysis_options.yaml")
        self.geom_dict = val.load_geometry_yaml(temp_geom)
        self.model_dict = val.load_modeling_yaml(temp_modopt)
        self.anal_dict = val.load_analysis_yaml(temp_analopt)

    def _on_demo_close(self):
        pass

    def _get_file_names(self):
        fname_geom = os.path.join(self.working_dir, f"{self.froot_export}_geometry.yaml")
        fname_model = os.path.join(self.working_dir, f"{self.froot_export}_modeling.yaml")
        fname_anal = os.path.join(self.working_dir, f"{self.froot_export}_analysis.yaml")
        return fname_geom, fname_model, fname_anal

    def _write_files(self):
        """
        The "Run WEIS" click event handler calls this method when it is ready
        to make an attempt to run WEIS. It checks to see if all file shave been
        edited
        """
        fname_geom, fname_model, fname_anal = self._get_file_names()
        val.write_yaml(self.geom_dict, fname_geom)
        val.write_yaml(self.model_dict, fname_model)
        val.write_yaml(self.anal_dict, fname_anal)

    def _run_program(self):
        fname_geom, fname_model, fname_anal = self._get_file_names()
        self._write_files()
        run_wisdem(fname_geom, fname_model, fname_anal)

    def _mode_set(self):
        if dpg.get_value("mode_dropdown") == "Input Editor":
            self._input_mode = True
        elif dpg.get_value("mode_dropdown") == "Output Viewer":
            self._input_mode = False
        else:
            print(dpg.get_value("mode_dropdown"), self._input_mode)
            raise ValueError("Shouldn't get here")

    def _set_workdir(self, sender, appdata):
        if not appdata is None:
            self.working_dir = appdata["current_path"].strip()
        dpg.configure_item("workdir_field", default_value=f"Working directory: {self.working_dir}")

    def _choose_workdir(self):
        dpg.add_file_dialog(directory_selector=True, show=True, width=500, height=400, callback=self._set_workdir)

    def _recursion_dict_display(self, dict_or_list, parent_tag=""):
        """
        This recursive method is where the automatic layout magic happens.
        This method calls itself recursively as it descends down the dictionary
        nesting structure.

        Basically, any given dictionary can consist of scalar and dictionary
        values. At each level of the dictionary, edit fields are placed for
        scalar values and tabbed widgets are placed for the next level of
        nesting.

        Parameters
        ----------
        dict_or_list: Dict[str, Any]
            The dictionary to automatically lay out in to the interface.
        """
        subscripts_values = dict_or_list.items() if type(dict_or_list) is dict else enumerate(dict_or_list)  # type: ignore

        for k, v in subscripts_values:
            # Create the unique tag for this item
            itag = dpg.generate_uuid()
            print(k, v)
            # Recursive call for nested dictionaries within dictionaries.
            if type(v) is dict:
                dpg.add_tree_node(label=k, tag=itag, parent=parent_tag)
                self._recursion_dict_display(v, parent_tag=itag)

            # Recursive call for nested dictionaries within lists.
            elif type(v) is list and len(v) > 0 and type(v[0]) is dict:
                dpg.add_tree_node(label=k, tag=itag, parent=parent_tag, bullet=True)
                self._recursion_dict_display(v, parent_tag=itag)

            # Otherwise just lay out a label and text field.
            else:
                DPGLineEdit(dict_or_list, k, v, parent_tag)

    def show_gui(self):
        def _log(sender, app_data, user_data):
            print(f"sender: {sender}, \t app_data: {app_data}, \t user_data: {user_data}")

        with dpg.window(width=800, height=800, on_close=self._on_demo_close, pos=(0, 0), no_title_bar=True):

            with dpg.menu_bar():

                with dpg.menu(label="Mode"):

                    dpg.add_combo(
                        ("Input Editor", "Output Viewer"),
                        width=140,
                        default_value="Input Editor",
                        label="Mode",
                        tag="mode_dropdown",
                        callback=self._mode_set,
                    )

                with dpg.menu(label="Files"):

                    dpg.add_menu_item(label="Set Working Diretory...", callback=self._choose_workdir)

                    with dpg.menu(label="Import"):

                        dpg.add_menu_item(label="Geometry YAML")
                        dpg.add_menu_item(label="Modeling YAML")
                        dpg.add_menu_item(label="Analysis YAML")
                        dpg.add_menu_item(label="Output NPZ")

                    dpg.add_menu_item(label="Export Input Files...")

                with dpg.menu(label="Run"):

                    dpg.add_menu_item(label="Run WISDEM/WEIS...", callback=self._run_program)

            # Display and/or change working directory
            with dpg.group(horizontal=True):
                dpg.add_text("Temp", tag="workdir_field")
                self._set_workdir(None, None)
                dpg.add_button(label="Change", callback=self._choose_workdir)

            dpg.add_text("WISDEM/WEIS YAML-Based Input Files:")

            with dpg.tab_bar():

                with dpg.tab(label="Geometry"):
                    with dpg.collapsing_header(label="WindIO Geometry", tag="geometry"):
                        self._recursion_dict_display(self.geom_dict, parent_tag="geometry")

                with dpg.tab(label="Modeling"):
                    with dpg.collapsing_header(label="WISDEM/WEIS Modeling", tag="modeling"):
                        self._recursion_dict_display(self.model_dict, parent_tag="modeling")

                with dpg.tab(label="Analysis"):
                    with dpg.collapsing_header(label="WISDEM/WEIS Analysis", tag="analysis"):
                        self._recursion_dict_display(self.anal_dict, parent_tag="analysis")

        """
            with tab
                with dpg.file_dialog(label="Demo File Dialog", width=300, height=400, show=False, callback=lambda s, a, u : print(s, a, u), tag="__demo_filedialog"):
                    dpg.add_file_extension(".*", color=(255, 255, 255, 255))
                    dpg.add_file_extension("Source files (*.cpp *.h *.hpp){.cpp,.h,.hpp}", color=(0, 255, 255, 255))
                    dpg.add_file_extension(".cpp", color=(255, 255, 0, 255))
                    dpg.add_file_extension(".h", color=(255, 0, 255, 255), custom_text="header")
                    dpg.add_file_extension("Python(.py){.py}", color=(0, 255, 0, 255))
                    #dpg.add_button(label="Button on file dialog")

                dpg.add_button(label="Show File Selector", user_data=dpg.last_container(), callback=lambda s, a, u: dpg.configure_item(u, show=True))


            dpg.add_button(label="Directory Selector", callback=lambda: dpg.show_item("file_dialog_id"))
        """


if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(title="NREL's WISDEM/WEIS Input/Output GUI", width=600, height=600)

    mygui = GUI_Master()
    mygui.show_gui()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
