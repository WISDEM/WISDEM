import os
import copy
import numpy as np
import jsonschema as json
import ruamel.yaml as ry
from functools import reduce
import operator


fschema_geom = os.path.join(os.path.dirname(os.path.realpath(__file__)), "geometry_schema.yaml")
fschema_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), "modeling_schema.yaml")
fschema_opt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "analysis_schema.yaml")


def load_yaml(fname_input : str) -> dict:
    """
    Reads and parses a YAML file in a safe mode using the ruamel.yaml library.

    Args:
        fname_input (str): Path to the YAML file to be loaded.

    Returns:
        dict: Parsed YAML content as a dictionary.
    """
    reader = ry.YAML(typ="safe", pure=True)
    with open(fname_input, "r", encoding="utf-8") as f:
        input_yaml = reader.load(f)
    return input_yaml

def write_yaml(instance : dict, foutput : str) -> None:
    """
    Writes a dictionary to a YAML file using the ruamel.yaml library.

    Args:
        instance (dict): Dictionary to be written to the YAML file.
        foutput (str): Path to the output YAML file.

    Returns:
        None
    """
    instance = remove_numpy(instance)

    # Write yaml with updated values
    yaml = ry.YAML()
    yaml.default_flow_style = None
    yaml.width = float("inf")
    yaml.indent(mapping=4, sequence=6, offset=3)
    yaml.allow_unicode = False
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)


# ---------------------
# This is for when the defaults are in another file
def nested_get(indict, keylist):
    rv = indict
    for k in keylist:
        rv = rv[k]
    return rv


def nested_set(indict, keylist, val):
    rv = indict
    for i, k in enumerate(keylist):
        rv = rv[k] if i != len(keylist) - 1 else val


def integrate_defaults(instance : dict, defaults : dict, yaml_schema : dict) -> dict:
    """
    Integrates default values from a dictionary into another dictionary.

    Args:
        instance (dict): Dictionary to be updated with default values.
        defaults (dict): Dictionary containing default values.
        yaml_schema (dict): Dictionary containing the schema of the YAML file.

    Returns:
        dict: Updated dictionary with default values integrated.
    """
    # Prep iterative validator
    # json.validate(self.wt_init, yaml_schema)
    validator = json.Draft7Validator(yaml_schema)
    errors = validator.iter_errors(instance)

    # Loop over errors
    for e in errors:
        # If the error is due to a missing required value, try to set it to the default
        if e.validator == "required":
            for k in e.validator_value:
                if k not in e.instance.keys():
                    mypath = e.absolute_path.copy()
                    mypath.append(k)
                    v = nested_get(defaults, mypath)
                    if isinstance(v, dict) or isinstance(v, list) or v in ["name", "material"]:
                        # Too complicated to just copy over default, so give it back to the user
                        raise (e)
                    print("WARNING: Missing value,", list(mypath), ", so setting to:", v)
                    nested_set(instance, mypath, v)
        raise (e)
    return instance


def simple_types(indict : dict) -> dict:
    """
    Recursively converts numpy array elements within a nested dictionary to lists and ensures
    all values are simple types (float, int, dict, bool, str).

    Args:
        indict (dict): The dictionary to process.

    Returns:
        dict: The processed dictionary with numpy arrays converted to lists and unsupported types to empty strings.
    """
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return simple_types(value)
        elif isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        elif isinstance(value, (float, int, bool, str)):
            return value
        else:
            return ""

    rv = {k: convert(v) for k, v in indict.items()}
    return rv


# ---------------------
# See: https://python-jsonschema.readthedocs.io/en/stable/faq/#why-doesn-t-my-schema-s-default-property-set-the-default-on-my-instance
def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return json.validators.extend(validator_class, {"properties": set_defaults})


DefaultValidatingDraft7Validator = extend_with_default(json.Draft7Validator)

def _validate(finput, fschema, defaults=True):
    """
    Validates a dictionary against a schema and returns the validated dictionary.

    Args:
        finput (dict or str): Dictionary or path to the YAML file to be validated.
        fschema (dict or str): Dictionary or path to the schema file to validate against.
        defaults (bool): Flag to indicate if default values should be integrated.

    Returns:
        dict: Validated dictionary.
    """
    schema_dict = fschema if isinstance(fschema, dict) else load_yaml(fschema)
    input_dict = finput if isinstance(finput, dict) else load_yaml(finput)
    validator = DefaultValidatingDraft7Validator if defaults else json.Draft7Validator
    validator(schema_dict).validate(input_dict)
    return input_dict


# ---------------------
def load_geometry_yaml(finput):
    return _validate(finput, fschema_geom)


def load_modeling_yaml(finput):
    return _validate(finput, fschema_model)


def load_analysis_yaml(finput):
    return _validate(finput, fschema_opt)


def write_geometry_yaml(instance : dict, foutput : str) -> None:
    _validate(instance, fschema_geom, defaults=False)
    # Ensure the file output has a .yaml suffix
    if not foutput.endswith(".yaml"):
        foutput += ".yaml"
    write_yaml(instance, foutput)


def write_modeling_yaml(instance : dict, foutput : str) -> None:
    _validate(instance, fschema_model, defaults=False)

    # Ensure the output filename does not end with .yaml or .yml
    if foutput.endswith(".yaml"):
        foutput = foutput[:-5]
    elif foutput.endswith(".yml"):
        foutput = foutput[:-4]
    sfx_str = "-modeling.yaml"

    instance2 = simple_types(instance)
    write_yaml(instance2, foutput + sfx_str)


def write_analysis_yaml(instance : dict, foutput : str) -> None:
    _validate(instance, fschema_opt, defaults=False)

    # Ensure the output filename does not end with .yaml or .yml
    if foutput.endswith(".yaml"):
        foutput = foutput[:-5]
    elif foutput.endswith(".yml"):
        foutput = foutput[:-4]

    sfx_str = "-analysis.yaml"
    write_yaml(instance, foutput + sfx_str)

def remove_numpy(fst_vt : dict) -> dict:
    """
    Recursively converts numpy array elements within a nested dictionary to lists and ensures
    all values are simple types (float, int, dict, bool, str) for writing to a YAML file.

    Args:
        fst_vt (dict): The dictionary to process.

    Returns:
        dict: The processed dictionary with numpy arrays converted to lists and unsupported types to simple types.
    """

    def get_dict(vartree, branch):
        return reduce(operator.getitem, branch, vartree)

    # Define conversion dictionary for numpy types
    conversions = {
        np.int_: int,
        np.intc: int,
        np.intp: int,
        np.int8: int,
        np.int16: int,
        np.int32: int,
        np.int64: int,
        np.uint8: int,
        np.uint16: int,
        np.uint32: int,
        np.uint64: int,
        np.single: float,
        np.double: float,
        np.longdouble: float,
        np.csingle: float,
        np.cdouble: float,
        np.float_: float,
        np.float16: float,
        np.float32: float,
        np.float64: float,
        np.complex64: float,
        np.complex128: float,
        np.bool_: bool,
        np.ndarray: lambda x: x.tolist(),
    }

    def loop_dict(vartree, branch):
        if not isinstance(vartree, dict):
            return fst_vt
        for var in vartree.keys():
            branch_i = copy.copy(branch)
            branch_i.append(var)
            if isinstance(vartree[var], dict):
                loop_dict(vartree[var], branch_i)
            else:
                current_value = get_dict(fst_vt, branch_i[:-1])[branch_i[-1]]
                data_type = type(current_value)
                if data_type in conversions:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = conversions[data_type](current_value)
                elif isinstance(current_value, (list, tuple)):
                    for i, item in enumerate(current_value):
                        current_value[i] = remove_numpy(item)

    # set fast variables to update values
    loop_dict(fst_vt, [])
    return fst_vt


if __name__ == "__main__":
    yaml_schema = load_yaml(fschema_opt)
    myobj = load_yaml("sample_analysis.yaml")
    DefaultValidatingDraft7Validator(yaml_schema).validate(myobj)
    # validator.validate( myobj )
    print([k for k in myobj.keys()])
    print(myobj["general"])

    obj = {}
    schema = {"properties": {"foo": {"default": "bar"}}}
    # Note jsonschem.validate(obj, schema, cls=DefaultValidatingDraft7Validator)
    # will not work because the metaschema contains `default` directives.
    DefaultValidatingDraft7Validator(schema).validate(obj)
    print(obj)
