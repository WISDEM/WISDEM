import os
import re
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np

try:
    import dearpygui.dearpygui as dpg
    GUI_AVAIL = True
except ImportError:
    GUI_AVAIL = False

import wisdem.inputs.validation as val
from wisdem.glue_code.runWISDEM import run_wisdem


def _hsv_to_rgb(h, s, v):
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)  # XXX assume int() truncates!
    f = (h * 6.0) - i
    p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0:
        return (255 * v, 255 * t, 255 * p)
    if i == 1:
        return (255 * q, 255 * v, 255 * p)
    if i == 2:
        return (255 * p, 255 * v, 255 * t)
    if i == 3:
        return (255 * p, 255 * q, 255 * v)
    if i == 4:
        return (255 * t, 255 * p, 255 * v)
    if i == 5:
        return (255 * v, 255 * p, 255 * q)
    

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

        elif self.is_boolean(value):
            dpg.add_input_text(
                label=key, default_value=str(bool(value)), width=150, callback=self.mycallback, parent=parent_tag
            )

        elif self.is_integer(value):
            # (have to list after is_boolean otherwise T/F becomes 1/0)
            dpg.add_input_int(
                label=key, default_value=int(value), width=200, callback=self.mycallback, parent=parent_tag
            )

        else:
            dpg.add_input_text(
                label=key, default_value=str(value), width=300, callback=self.mycallback, parent=parent_tag
            )

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

        self.working_dir = os.path.expanduser("~")
        self.froot_export = "gui_export"

        self.geometry_dict = {}
        self.modeling_dict = {}
        self.analysis_dict = {}

        self.output_data = None
        
    def _on_demo_close(self):
        pass

    def _get_file_names(self):
        fname_geom = f"{self.froot_export}_geometry.yaml"
        fname_model = f"{self.froot_export}_modeling.yaml"
        fname_anal = f"{self.froot_export}_analysis.yaml"
        return fname_geom, fname_model, fname_anal

    def _write_files(self):
        """
        The "Run WEIS" click event handler calls this method when it is ready
        to make an attempt to run WEIS. It checks to see if all file shave been
        edited
        """
        fname_geom, fname_model, fname_anal = self._get_file_names()
        val.write_yaml(self.geometry_dict, fname_geom)
        val.write_yaml(self.modeling_dict, fname_model)
        val.write_yaml(self.analysis_dict, fname_anal)

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

    def _run_program(self):
        with dpg.window(modal=True, no_close=True, popup=True, show=True, autosize=True, tag="run_popup"):
            dpg.add_text("Running WISDEM in terminal or console window")

            current_dir = os.getcwd()
            os.chdir(self.working_dir)

            fname_geom, fname_model, fname_anal = self._get_file_names()
            self._write_files()
            run_wisdem(fname_geom, fname_model, fname_anal)

            os.chdir(current_dir)

        dpg.configure_item("run_popup", show=False)

    def _export_root(self, sender, app_data):
        self.froot_export = app_data

    def _read_output(self, sender, app_data):
        if app_data is not None:
            # Grab the filename selected
            temp_dict = app_data["selections"]
            fname = [temp_dict[m] for m in temp_dict][0]

            # Read in the file
            if fname.lower().endswith('.pkl'):
                temp_data = pd.read_pickle(fname)
            elif fname.lower().endswith('.csv'):
                temp_data = pd.read_csv(fname)
            elif fname.lower().endswith('.xlsx'):
                temp_data = pd.read_excel(fname)

            # Fancy way of dropping openmdao duplicates
            self.output_data = temp_data.loc[temp_data.astype(str).drop_duplicates(subset=['units','values']).index]

            # Loop over all data and cleanup
            temp_size = np.zeros(len(self.output_data), dtype=np.int_)
            temp_val = [0]*len(self.output_data)
            var_full  = ['']*len(self.output_data)
            to_delete = []
            for k in range(len(self.output_data)):
                ivar = self.output_data['variables'].iloc[k]
                ival = self.output_data['values'].iloc[k]
                iunit = self.output_data['units'].iloc[k]

                # csv and xlsx bring in values as string- convert to python types
                if isinstance(ival, str):
                    try:
                        temp_val[k] = eval(ival)
                        ival = temp_val[k]
                    except:
                        to_delete.append(self.output_data.index[k])
                else:
                    temp_val[k] = ival

                # Cleanup no-unit entries
                if str(iunit).lower() in ['','na','n/a','none','nan']:
                    iunit = '-'

                # Identify size
                try:
                    temp_size[k] = len(ival)
                except:
                    temp_size[k] = 1

                # Variable axis/listbox name
                var_full[k] = f'{ivar} [{iunit}] ({temp_size[k]})'

            # Store clean data
            self.output_data['size'] = temp_size
            self.output_data['variables_full'] = var_full
            self.output_data['values'] = temp_val

            # Drop bad data, then sort
            self.output_data.drop(index=to_delete, inplace=True)
            self.output_data.sort_values('variables_full', inplace=True)

            # Update list box
            dpg.configure_item("xylistlabel", show=True)
            dpg.configure_item("xylist", show=True)
            dpg.configure_item("plotbutton", show=True)
            dpg.configure_item("xlist", items=self.output_data['variables_full'].to_list(), show=True)
            dpg.configure_item("ylist", items=self.output_data['variables_full'].to_list(), show=True)
            
    def _load_output(self):
        with dpg.file_dialog(label="WISDEM Output File Selector", directory_selector=False, show=True,
                             callback=self._read_output, width=500, height=400, default_path=self.working_dir):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255), custom_text="[Excel]")
            dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[CSV]")
            dpg.add_file_extension(".pkl", color=(0, 255, 0, 255), custom_text="[Pickle]")

    def _set_workdir(self, sender, app_data):
        if app_data is not None:
            self.working_dir = app_data["current_path"].strip()
        dpg.configure_item("workdir_field", default_value=f"Working directory: {self.working_dir}")

    def _choose_workdir(self):
        dpg.add_file_dialog(
            label="Working Directory Selector",
            directory_selector=True,
            show=True,
            width=500,
            height=400,
            callback=self._set_workdir,
            default_path=self.working_dir,
        )

    def _set_file_field(self, sender, app_data, user_data):
        if not app_data is None:
            fpath = app_data["file_path_name"].strip()

            if user_data.lower().find("geometry") >= 0:
                self.geometry_dict = val.load_geometry_yaml(fpath)
                mydict = self.geometry_dict
                id_root = "geometry"

            elif user_data.lower().find("modeling") >= 0:
                self.modeling_dict = val.load_modeling_yaml(fpath)
                mydict = self.modeling_dict
                id_root = "modeling"

            elif user_data.lower().find("analysis") >= 0:
                self.analysis_dict = val.load_analysis_yaml(fpath)
                mydict = self.analysis_dict
                id_root = "analysis"

        # Delete the old nested yaml file and add the new one
        obj_id = id_root + "_context"  # user_data.replace("field", "context")
        dpg.delete_item(obj_id, children_only=True)
        self._recursion_dict_display(mydict, parent_tag=obj_id)

    def _choose_file(self, sender, app_data, user_data):
        with dpg.file_dialog(
            label="YAML File Selector",
            directory_selector=False,
            show=True,
            width=500,
            height=400,
            user_data=user_data,
            callback=self._set_file_field,
        ):
            dpg.add_file_extension("YAML Files (*.yml *.yaml){.yml,.yaml}", color=(0, 255, 255, 255))
            dpg.add_file_extension(".*", color=(255, 255, 255, 255))

    def _variable_list(self, sender, app_data, user_data):
        # When enter x-value in listbox, reduce y-values to equivalent size kosher for plotting
        xval_full = dpg.get_value("xselected")
        xdf = self.output_data[ self.output_data['variables_full'] == xval_full ]
        xsize = int(xdf['size'].iloc[0])
        yreduced = self.output_data[ self.output_data['size']==xsize ]
        dpg.configure_item("ylist", items=yreduced['variables_full'].to_list(), show=True)

    def _gen_plot(self, sender, app_data, user_data):
        # Build the plot, reset the axis
        xval_full = dpg.get_value("xselected")
        yval_full = dpg.get_value("yselected")
        xstr = " ".join(xval_full.split()[:-1])
        ystr = " ".join(yval_full.split()[:-1])
        xdf = self.output_data[ self.output_data['variables_full'] == xval_full ]
        ydf = self.output_data[ self.output_data['variables_full'] == yval_full ]
        xx = np.array(xdf['values'].to_list()).flatten()
        yy = np.array(ydf['values'].to_list()).flatten()
        dpg.configure_item("xaxis", label=xstr)
        dpg.configure_item("yaxis", label=ystr)
        dpg.configure_item("lineseries", x=xx, y=yy)
        dpg.reset_axis_zoom_constraints('xaxis')
        dpg.reset_axis_zoom_constraints('yaxis')
        dpg.reset_axis_limits_constraints('xaxis')
        dpg.reset_axis_limits_constraints('yaxis')
        dpg.fit_axis_data('xaxis')
        dpg.fit_axis_data('yaxis')
        dpg.reset_axis_ticks('xaxis')
        dpg.reset_axis_ticks('yaxis')
        dpg.configure_item("myplot", show=True)
                      
    
    def show_gui(self):
        def _log(sender, app_data, user_data):
            print(f"sender: {sender}, \t app_data: {app_data}, \t user_data: {user_data}")

        # GUI variables for selectors
        with dpg.value_registry():
            dpg.add_string_value(default_value="x-axis string", tag="xselected")
            dpg.add_string_value(default_value="y-axis string", tag="yselected")
            
        # Make buttons stand out a big more
        with dpg.theme(tag="button_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, _hsv_to_rgb(3.0 / 7.0, 0.6, 0.6))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, _hsv_to_rgb(3.0 / 7.0, 0.8, 0.8))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, _hsv_to_rgb(3.0 / 7.0, 0.7, 0.7))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

        # INPUT WINDOW
        with dpg.window(width=750, height=800, on_close=self._on_demo_close, pos=(0, 0), label='Input'):
            with dpg.menu_bar():
                #with dpg.menu(label="Mode"):
                #    dpg.add_combo( ("Input Editor", "Output Viewer"), width=140, default_value="Input Editor",
                #                   label="Mode", tag="mode_dropdown", callback=self._mode_set)

                with dpg.menu(label="Files"):
                    dpg.add_menu_item(label="Set Working Diretory...", callback=self._choose_workdir)

                    with dpg.menu(label="Import"):
                        dpg.add_menu_item(label="Geometry YAML", user_data="geometry", callback=self._choose_file)
                        dpg.add_menu_item(label="Modeling YAML", user_data="modeling", callback=self._choose_file)
                        dpg.add_menu_item(label="Analysis YAML", user_data="analysis", callback=self._choose_file)
                        # dpg.add_menu_item(label="Output NPZ")

                    dpg.add_menu_item(label="Export Input Files...")

                with dpg.menu(label="Run"):
                    dpg.add_menu_item(label="Run WISDEM/WEIS...", callback=self._run_program)

            # Display and/or change working directory
            dpg.add_text("")
            
            with dpg.group(horizontal=True):
                dpg.add_text("Temp", tag="workdir_field")
                dpg.add_button(label="Change", callback=self._choose_workdir)
                dpg.bind_item_theme(dpg.last_item(), "button_theme")

            with dpg.group(horizontal=True):
                dpg.add_text("Export Prefix:")
                dpg.add_input_text(label="", default_value=self.froot_export, width=200, callback=self._export_root, tag="export_field")
                
            dpg.add_text("")
            dpg.add_text("WISDEM/WEIS YAML-Based Input Files:", color=(255, 255, 0))

            with dpg.tab_bar():
                with dpg.tab(label="Geometry"):
                    dpg.add_button(label="Import YAML File", user_data="geometry", callback=self._choose_file)
                    dpg.bind_item_theme(dpg.last_item(), "button_theme")
                    dpg.add_text("")
                    dpg.add_collapsing_header(label="WindIO Geometry", tag="geometry_context")

                with dpg.tab(label="Modeling"):
                    dpg.add_button(label="Import YAML File", user_data="modeling", callback=self._choose_file)
                    dpg.bind_item_theme(dpg.last_item(), "button_theme")
                    dpg.add_text("")
                    dpg.add_collapsing_header(label="WISDEM/WEIS Modeling", tag="modeling_context")

                with dpg.tab(label="Analysis"):
                    dpg.add_button(label="Import YAML File", user_data="analysis", callback=self._choose_file)
                    dpg.bind_item_theme(dpg.last_item(), "button_theme")
                    dpg.add_text("")
                    dpg.add_collapsing_header(label="WISDEM/WEIS Analysis", tag="analysis_context")

        # Do init steps
        temp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
            "examples",
            "02_reference_turbines",
        )
        temp_geom = os.path.join(temp_dir, "IEA-15-240-RWT.yaml")
        temp_model = os.path.join(temp_dir, "modeling_options.yaml")
        temp_anal = os.path.join(temp_dir, "analysis_options.yaml")

        self._set_workdir(None, None)
        self._set_file_field(None, {"file_path_name": temp_geom}, "geometry_field")
        self._set_file_field(None, {"file_path_name": temp_model}, "modeling_field")
        self._set_file_field(None, {"file_path_name": temp_anal}, "analysis_field")

        # OUTPUT WINDOW
        with dpg.window(width=900, height=800, on_close=self._on_demo_close, pos=(750, 0), label='Output'):
            
            with dpg.menu_bar():
                with dpg.menu(label="Load"):
                    dpg.add_menu_item(label="Select output file", callback=self._load_output)

            with dpg.group(horizontal=True, tag='xylistlabel', show=False):
                dpg.add_text("X-Axis Variable")
                dpg.add_text(" "*50)
                dpg.add_text("Y-Axis Variable")
                
            with dpg.group(horizontal=True, tag='xylist', show=False):
                dpg.add_listbox([], label='', tag='xlist', width=450, source="xselected", num_items=15, callback=self._variable_list)
                dpg.add_listbox([], label='', tag='ylist', width=450, source="yselected", num_items=15)

            with dpg.group(horizontal=True, tag='plotbutton', show=False):
                dpg.add_button(label="Line Plot", callback=self._gen_plot, user_data="line")
                dpg.bind_item_theme(dpg.last_item(), "button_theme")
                #dpg.add_button(label="Bar Plot", callback=self._gen_plot, user_data="bar")
                #dpg.bind_item_theme(dpg.last_item(), "button_theme")

            with dpg.plot(label="WISDEM Plot", height=400, width=-1, show=False, tag='myplot'):
                #dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag='xaxis')
                with dpg.plot_axis(dpg.mvYAxis, label="y", tag='yaxis'):
                    # GUI crashes if these are numpy arrays, but lists seem to work.  Don't know why.
                    dpg.add_line_series(np.arange(10).tolist(), np.arange(10).tolist(), parent='yaxis', tag='lineseries')


def run():
    if GUI_AVAIL:
        dpg.create_context()
        dpg.create_viewport(title="NREL's WISDEM/WEIS Input/Output GUI", width=1650, height=800)

        mygui = GUI_Master()
        mygui.show_gui()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
    else:
        print("GUI is not available")

if __name__ == "__main__":
    run()
