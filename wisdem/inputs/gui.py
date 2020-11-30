from typing import Dict, Any, List, Union
import yaml
import sys, traceback
import re
from pathlib import Path

from PySide2.QtWidgets import (  # type: ignore
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QFormLayout,
    QTabWidget,
    QWidget,
    QMainWindow,
    QFileDialog,
    QApplication,
    QMessageBox,
)

from wisdem.glue_code.runWISDEM import run_wisdem


class FocusQLineEdit(QLineEdit):
    """
    FocusQLineEdit subclasses QLineEdit to add the following functionality:

    - Observing focus loss events and, when focus is lost

    - Updating a value in given dictionary and key based on the input
      in this text field.
    """

    def __init__(self, *args, **kwargs):
        """
        The instance attributes for this class are:

        _dictionary: Dict[str, Any]
            The dictionary to be changed when this text field is changed.

        _key_on_dictionary: str
            The key on the dictionary to be changed when this text field
            changes.

        _list_re: re.Pattern
            The regex that matches a string that contains a list, so that
            it does not need to be instantiated each time it is used.

        These attributes default to None. If either one of them is None, no
        attempt will be made to change the dictionary.
        """
        self._dictionary = None
        self._key_on_dictionary = None
        self._list_re = re.compile(r"^\[.*\]$")
        super(FocusQLineEdit, self).__init__(*args, *kwargs)

    def set_dictionary_and_key(self, dictionary: Union[List[Any], Dict[str, Any]], key: str) -> None:
        """
        This method sets the dictionary and key to be modified when the focus
        changes out of this widget

        Parameters
        ----------
        dictionary: Dict[str, Any]
            The dictionary to be changed.

        key: str
            The key whose value is to be changed on the dictionary.
        """
        self._dictionary = dictionary
        self._key_on_dictionary = key

    def focusOutEvent(self, arg__1) -> None:
        """
        Overrides focusOutEvent() in the base class (the base class's method is
        called before anything in this method executes, though).

        The purpose of this override is to observe focus out events. When this
        control looses focus, it updates the underlying specified in the
        instance attributes.
        """
        super(FocusQLineEdit, self).focusOutEvent(arg__1)
        if self._dictionary is not None and self._key_on_dictionary is not None:
            if self.is_list(self.text()):
                value = self.parse_list()
            elif self.is_float(self.text()):
                value = float(self.text())  # type: ignore
            elif self.is_boolean(self.text()):
                value = self.parse_boolean(self.text())  # type: ignore
            else:
                value = self.text()  # type: ignore
            self._dictionary[self._key_on_dictionary] = value
            # print("New level dictionary", self._dictionary)
        # else:
        #     print("Focus lost, but dictionary and key are not set.")

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
        if value == "True" or value == "true":
            return True
        else:
            return False

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
        return value == "True" or value == "False" or value == "true" or value == "false"


class FormAndMenuWindow(QMainWindow):
    """
    This class creates a form to edit a dictionary. It nests tabs for different
    levels of the nesting within the dictionaries.

    This automatically builds an interface from the dictionary.
    """

    def __init__(self, parent=None):
        """
        Parameters
        ----------
        dict_to_edit: Dict[str, Any]
            The dictionary to edit in this form.

        output_filename: str
            The filename to write the dictionary when the save button is clicked.

        parent
            The parent as needed by the base class.
        """
        super(FormAndMenuWindow, self).__init__(parent)
        self.analysis_yaml_editor_widget = None
        self.modeling_yaml_widget = None
        self.geometry_yaml_widget = None
        self.geometry_filename_line_edit = None
        self.modeling_filename_line_edit = None
        self.analysis_filename_line_edit = None
        self.geometry_dict = None
        self.analysis_dict = None
        self.modeling_dict = None
        self.geometry_filename = None
        self.analysis_filename = None
        self.modeling_filename = None
        self.main_widget = None
        self.status_widget = None
        self.status_label = None

    def setup(self) -> None:
        """
        After this class is instantiated, this method should be called to
        lay out the user interface.
        """
        self.setWindowTitle("YAML GUI")
        # self.setup_menu_bar()

        central_widget = self.create_central_widget()
        self.setCentralWidget(central_widget)

    def recursion_ui_setup(self, dict_or_list: Union[List[Any], Dict[str, Any]]) -> QFormLayout:
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
        form_level_layout = QFormLayout()
        dict_tabs = QTabWidget()
        display_tabs = False
        subscripts_values = dict_or_list.items() if type(dict_or_list) is dict else enumerate(dict_or_list)  # type: ignore
        for k, v in subscripts_values:

            # Recursive call for nested dictionaries within dictionaries.
            if type(v) is dict:
                display_tabs = True
                child_widget = QWidget()
                child_layout = self.recursion_ui_setup(v)
                child_widget.setLayout(child_layout)
                dict_tabs.addTab(child_widget, str(k))

            # Recursive call for nested dictionaries within lists.
            elif type(v) is list and type(v[0]) is dict:
                display_tabs = True
                child_widget = QWidget()
                child_layout = self.recursion_ui_setup(v)
                child_widget.setLayout(child_layout)
                dict_tabs.addTab(child_widget, str(k))

            # Otherwise just lay out a label and text field.
            else:
                line_edit = FocusQLineEdit(str(v))
                line_edit.setMinimumWidth(150)
                line_edit.set_dictionary_and_key(dict_or_list, k)
                form_level_layout.addRow(QLabel(k), line_edit)

        # If there is a nested dictionary, display it.
        if display_tabs:
            form_level_layout.addRow(dict_tabs)

        # Return the whole layout
        return form_level_layout

    def write_dict_to_yaml(self) -> None:
        """
        This is the event handler for the save button. It simply writes the
        dictionary (which has been continuously updated during focus out
        events) to a YAML file as specified in self.output_filename.
        """
        with open(self.output_filename, "w") as file:
            yaml.dump(self.dict_to_edit, file)

    def create_central_widget(self) -> QWidget:
        """
        Returns
        -------
        QWidget
            The form with buttons on it.
        """
        status_and_main_widget = QWidget()
        status_and_main_widget_layout = QVBoxLayout()

        self.status_widget = QWidget()
        status_form_layout = QFormLayout()
        status_label = QLabel("Status:")
        self.status_label = QLabel()
        self.status_label.setText("Please create simulation configurations.")
        status_form_layout.addRow(status_label, self.status_label)
        self.status_widget.setLayout(status_form_layout)

        self.main_widget = QWidget()
        subsection_width = 500
        subsection_height = 900

        geometry_section_label = QLabel("Geometry")
        geometry_section_label.setStyleSheet("font-weight: bold;")
        geometry_filename_button = QPushButton("Select geometry YAML...")
        # geometry_visualize_button = QPushButton("Visualize geometry")
        self.geometry_filename_line_edit = QLineEdit()
        self.geometry_filename_line_edit.setPlaceholderText("Please select a geometry file.")
        self.geometry_filename_line_edit.setFixedWidth(200)
        self.geometry_filename_line_edit.setReadOnly(True)
        geometry_filename_button.clicked.connect(self.file_picker_geometry)

        modeling_section_label = QLabel("Modeling")
        modeling_section_label.setStyleSheet("font-weight: bold;")
        modeling_filename_button = QPushButton("Select modeling YAML...")
        self.modeling_filename_line_edit = QLineEdit()
        self.modeling_filename_line_edit.setPlaceholderText("Please select a modeling file.")
        self.modeling_filename_line_edit.setFixedWidth(200)
        self.modeling_filename_line_edit.setReadOnly(True)
        modeling_filename_button.clicked.connect(self.file_picker_modeling)

        analysis_section_label = QLabel("Analysis")
        analysis_section_label.setStyleSheet("font-weight: bold;")
        analysis_filename_button = QPushButton("Select analysis YAML...")
        self.analysis_filename_line_edit = QLineEdit()
        self.analysis_filename_line_edit.setPlaceholderText("Please select an analysis file...")
        self.analysis_filename_line_edit.setFixedWidth(200)
        self.analysis_filename_line_edit.setReadOnly(True)
        analysis_filename_button.clicked.connect(self.file_picker_analysis)

        run_weis_button = QPushButton("Run WISDEM")
        run_weis_button.clicked.connect(self.run_weis_clicked)

        self.modeling_yaml_widget = QWidget()
        self.analysis_yaml_editor_widget = QWidget()
        self.geometry_yaml_widget = QWidget()

        geometry_layout = QFormLayout()
        geometry_layout.addRow(geometry_section_label)
        geometry_layout.addRow(self.geometry_filename_line_edit, geometry_filename_button)
        # geometry_layout.addRow(geometry_visualize_button)
        geometry_layout.addRow(self.geometry_yaml_widget)
        geometry_widget = QWidget()
        geometry_widget.setFixedWidth(subsection_width)
        geometry_widget.setFixedHeight(subsection_height)
        geometry_widget.setLayout(geometry_layout)

        modeling_layout = QFormLayout()
        modeling_layout.addRow(modeling_section_label)
        modeling_layout.addRow(self.modeling_filename_line_edit, modeling_filename_button)
        modeling_layout.addRow(self.modeling_yaml_widget)
        modeling_widget = QWidget()
        modeling_widget.setFixedWidth(subsection_width)
        modeling_widget.setFixedHeight(subsection_height)
        modeling_widget.setLayout(modeling_layout)

        analysis_layout = QFormLayout()
        analysis_layout.addRow(analysis_section_label)
        analysis_layout.addRow(self.analysis_filename_line_edit, analysis_filename_button)
        analysis_layout.addRow(self.analysis_yaml_editor_widget)
        analysis_widget = QWidget()
        analysis_widget.setFixedWidth(subsection_width)
        analysis_widget.setFixedHeight(subsection_height)
        analysis_widget.setLayout(analysis_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(geometry_widget)
        main_layout.addWidget(modeling_widget)
        main_layout.addWidget(analysis_widget)
        main_layout.addWidget(run_weis_button)

        self.main_widget.setLayout(main_layout)
        # return self.main_widget

        status_and_main_widget_layout.addWidget(self.status_widget)
        status_and_main_widget_layout.addWidget(self.main_widget)
        status_and_main_widget.setLayout(status_and_main_widget_layout)

        return status_and_main_widget

    def run_weis_clicked(self):
        """
        When the "Run WEIS" button is clicked, a popup dialog
        will pop up asking what the user wants to do. If the user
        has skipped any configuration files, then it will tell them
        to go back and create those configuration files. If the user
        has specified all the configuration files, then it will prompt
        the user for confirmation before saving the WEIS configuration
        files.
        """
        if self.geometry_filename is None:
            msg = QMessageBox()
            msg.setText("Run WISDEM: Missing file")
            msg.setInformativeText("You did not specify a geometry file.")
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
        elif self.modeling_filename is None:
            msg = QMessageBox()
            msg.setText("Run WISDEM: Missing file")
            msg.setInformativeText("You did not specify a modeling file.")
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
        elif self.analysis_filename is None:
            msg = QMessageBox()
            msg.setText("Run WISDEM: Missing file")
            msg.setInformativeText("You did not specify an analysis file.")
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
        else:
            self.status_label.setText("Writing files...")
            self.write_configuration_files()
            self.status_label.setText("Configuration files written.")
            msg = QMessageBox()
            msg.setText("Run WISDEM: Configuration files complete!")
            msg.setInformativeText("Click cancel to back out and continue editing. Click OK to run WISDEM.")
            msg.addButton(QMessageBox.Cancel)
            msg.addButton(QMessageBox.Ok)
            choice = msg.exec()
            if choice == QMessageBox.Ok:
                self.disable_ui_and_execute_wisdem()

    def write_configuration_files(self):
        """
        The "Run WEIS" click event handler calls this method when it is ready
        to make an attempt to run WEIS. It checks to see if all file shave been
        edited
        """
        if self.geometry_filename is not None:
            print(f"Writing geometry: {self.geometry_filename}")
            with open(self.geometry_filename, "w") as file:
                yaml.dump(self.geometry_dict, file)
        else:
            print("No geometry file to write")

        if self.analysis_filename is not None:
            print(f"Writing analysis: {self.analysis_filename}")
            with open(self.analysis_filename, "w") as file:
                yaml.dump(self.analysis_dict, file)
        else:
            print("No analysis file to write")

        if self.modeling_filename is not None:
            print(f"Writing modeling: {self.modeling_filename}")
            with open(self.modeling_filename, "w") as file:
                yaml.dump(self.modeling_dict, file)
        else:
            print("No modeling file to write")

    def disable_ui_and_execute_wisdem(self):
        """
        This method disables all widgets on the UI and executes WISDEM. It
        displays message boxes depending on whether WISDEM executed
        successfully.
        """
        self.main_widget.setEnabled(False)
        self.status_label.setText("Running WISDEM")

        try:
            wt_opt, modeling_options, analysis_options = run_wisdem(
                self.geometry_filename, self.modeling_filename, self.analysis_filename
            )
        except Exception as err:
            short_error_message = f"{type(err)}: {err}. More details on command line."
            traceback.print_exc(file=sys.stdout)
            self.status_label.setText("Execution error")
            msg = QMessageBox()
            msg.setText("WISDEM execution error")
            msg.setInformativeText(short_error_message)
            msg.addButton(QMessageBox.Ok)
            msg.exec()
            self.main_widget.setEnabled(True)
        else:
            self.status_label.setText("Execution success")
            msg = QMessageBox()
            msg.setText("WISDEM executed successfully")
            msg.addButton(QMessageBox.Ok)
            msg.exec()
            self.main_widget.setEnabled(True)

    def file_picker_geometry(self):
        """
        Shows the open file dialog for the geometry file.

        Returns
        -------
        None
            Returns nothing for now.
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        self.geometry_filename, _ = dialog.getOpenFileName(None, "Open File", str(Path.home()), "YAML (*.yml *.yaml)")
        self.geometry_filename_line_edit.setText(self.geometry_filename)
        self.geometry_dict = self.read_yaml_to_dictionary(self.geometry_filename)
        layout = self.recursion_ui_setup(self.geometry_dict)
        self.geometry_yaml_widget.setLayout(layout)

    def file_picker_modeling(self):
        """
        Shows the open dialog

        Returns
        -------
        None
            Returns nothing for now.
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        self.modeling_filename, _ = dialog.getOpenFileName(None, "Open File", str(Path.home()), "YAML (*.yml *.yaml)")
        self.modeling_filename_line_edit.setText(self.modeling_filename)
        self.modeling_dict = self.read_yaml_to_dictionary(self.modeling_filename)
        layout = self.recursion_ui_setup(self.modeling_dict)
        self.modeling_yaml_widget.setLayout(layout)

    def file_picker_analysis(self) -> None:
        """
        Shows the open dialog for the analysis YAML

        Returns
        -------
        None
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        self.analysis_filename, _ = dialog.getOpenFileName(None, "Open File", str(Path.home()), "YAML (*.yml *.yaml)")
        self.analysis_filename_line_edit.setText(self.analysis_filename)
        self.analysis_dict = self.read_yaml_to_dictionary(self.analysis_filename)
        layout = self.recursion_ui_setup(self.analysis_dict)
        self.analysis_yaml_editor_widget.setLayout(layout)

    @staticmethod
    def read_yaml_to_dictionary(input_filename: str) -> Dict[str, Any]:
        """
        This reads the YAML input which is used to build the user interface.
        """
        with open(input_filename) as file:
            result = yaml.load(file, Loader=yaml.FullLoader)
        return result


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    form = FormAndMenuWindow()
    form.setup()
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
