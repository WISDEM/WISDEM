import os
import copy
import numpy as np
import jsonschema as json
import jsonmerge
from functools import reduce
import operator
from openmdao.utils.mpi import MPI
import windIO.schemas as windio
from windIO.yaml import load_yaml, write_yaml
from windIO.validator import _enforce_no_additional_properties, _jsonschema_validate_modified
from pathlib import Path
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource

fschema_windio = os.path.join(os.path.dirname(os.path.realpath(windio.__file__)), "turbine", "turbine_schema.yaml")
fschema_geom = os.path.join(os.path.dirname(os.path.realpath(__file__)), "geometry_schema.yaml")
fschema_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), "modeling_schema.yaml")
fschema_opt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "analysis_schema.yaml")

schemaPath = Path(__file__).parent

def retrieve_yaml(uri: str):
    if not uri.endswith(".yaml"):
        raise NoSuchResource(ref=uri)
    path = schemaPath / Path(uri)
    contents = load_yaml(path)
    return Resource.from_contents(contents)


registry = Registry(retrieve=retrieve_yaml)

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
            return convert(value.tolist())
        elif isinstance(value, dict):
            return {key: convert(value) for key, value in value.items()}
        elif isinstance(value, (list, tuple, set)):
            return [convert(item) for item in value]  # treat all as list
        elif isinstance(value, (np.generic)):
            return value.item()  # convert numpy primatives to python primative underlying
        elif isinstance(value, (float, int, bool, str)):
            return value  # this should be the end case
        else:
            return ""
    return convert(indict)


# ---------------------
def extend_with_default(validator_class):
    # https://python-jsonschema.readthedocs.io/en/stable/faq/#why-doesn-t-my-schema-s-default-property-set-the-default-on-my-instance
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return json.validators.extend(validator_class, {"properties": set_defaults})

def extend_remove_additional(validator_class):
    # https://stackoverflow.com/questions/44694835/remove-properties-from-json-object-not-present-in-schema
    validate_properties = validator_class.VALIDATORS["properties"]

    def remove_additional_properties(validator, properties, instance, schema):
        for prop in list(instance.keys()):
            if prop not in properties:
                del instance[prop]

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return json.validators.extend(validator_class, {"properties" : remove_additional_properties})

DefaultValidatingDraft7Validator = extend_with_default(json.Draft7Validator)
RemovalValidatingDraft7Validator = extend_remove_additional(json.Draft7Validator)

def MPI_load_yaml(fname):
    """
    When MPI is active, loads a yaml on rank 0 and broadcasts it out

    Args:
        fname: file name of input yaml file

    Returns:
        dict: Dictionary corresponding to that yaml file
    """

    rank = MPI.COMM_WORLD.Get_rank()
    dict_yaml = load_yaml(fname) if rank == 0 else None
    dict_yaml = MPI.COMM_WORLD.bcast(dict_yaml, root = 0)

    return dict_yaml

def _validate(finput, fschema, defaults=True, removal=False, restrictive=False, rank_0 = False):
    """
    Validates a dictionary against a schema and returns the validated dictionary.

    Args:
        finput (dict or str): Dictionary or path to the YAML file to be validated.
        fschema (dict or str): Dictionary or path to the schema file to validate against.
        defaults (bool, optional): Flag to indicate if default values should be integrated.
        removal (bool, optional): Flag to indicate if entries outside of the schema should be removed
        restrictive (bool, optional): Flag to indicate if strict adherence to schema (no additions)
        rank_0 (bool, optional): Flag that should be set to true when the _validate function is
        called with MPI turned on and rank=0. Otherwise it can be kept as default to False

    Returns:
        dict: Validated dictionary.
    """
    # Read schema as dictionary
    if isinstance(fschema, dict):
        schema_dict = fschema
    else:
        schema_dict = MPI_load_yaml(fschema) if (MPI and rank_0 == False) else load_yaml(fschema)
        
    if restrictive:
        schema_dict = _enforce_no_additional_properties(schema_dict)

    # Read input file as dictionary
    if isinstance(finput, dict):
        input_dict = finput
    else:
        input_dict = MPI_load_yaml(finput) if (MPI and rank_0 == False) else load_yaml(finput)

    # Deep copy to ensure no shared references from yaml pointers and anchors
    unique_input_dict = deep_copy_without_shared_refs(input_dict)

    # WindIO way
    if defaults:
        _jsonschema_validate_modified(unique_input_dict, schema_dict, cls=DefaultValidatingDraft7Validator, registry=registry)
    elif removal:
        _jsonschema_validate_modified(unique_input_dict, schema_dict, cls=RemovalValidatingDraft7Validator, registry=registry)
    else:
        _jsonschema_validate_modified(unique_input_dict, schema_dict, registry=registry)

    
    # New deep copy to ensure no shared references from yaml pointers and anchors
    unique_input_dict = deep_copy_without_shared_refs(unique_input_dict)
    
    # Old way
    #validator = DefaultValidatingDraft7Validator if defaults else json.Draft7Validator
    #validator(schema_dict).validate(unique_input_dict)

    return unique_input_dict

def deep_copy_without_shared_refs(obj):
    """
    Recursively creates a deep copy of the object, breaking any shared references.
    """
    if isinstance(obj, dict):
        return {k: deep_copy_without_shared_refs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_copy_without_shared_refs(item) for item in obj]
    else:
        return obj  # Return the value directly if not a container

# ---------------------
def get_geometry_schema():
    windio_schema = load_yaml(fschema_windio)
    wisdem_schema = load_yaml(fschema_geom)
    merged_schema = jsonmerge.merge(windio_schema, wisdem_schema)
    return merged_schema

def load_geometry_yaml(finput):
    merged_schema = get_geometry_schema()
    return _validate(finput, merged_schema, restrictive=False) #True)


def load_modeling_yaml(finput):
    return _validate(finput, fschema_model, restrictive=True)


def load_analysis_yaml(finput):
    return _validate(finput, fschema_opt, restrictive=True)


def write_geometry_yaml(instance, foutput):
    merged_schema = get_geometry_schema()
    _validate(instance, merged_schema, restrictive=False, removal=False, defaults=False)
    sfx_str = '.yaml'
    if foutput[-5:] == sfx_str:
        sfx_str = ''
    write_yaml(instance, foutput+sfx_str)

    
def write_modeling_yaml(instance : dict, foutput : str) -> None:
    _validate(instance, fschema_model, restrictive=True, removal=True, defaults=False, rank_0=True)

    # Ensure the output filename does not end with .yaml or .yml
    if foutput.endswith(".yaml"):
        foutput = foutput[:-5]
    elif foutput.endswith(".yml"):
        foutput = foutput[:-4]
    sfx_str = "-modeling.yaml"

    instance2 = simple_types(instance)
    write_yaml(instance2, foutput + sfx_str)
    return foutput + sfx_str


def write_analysis_yaml(instance : dict, foutput : str) -> None:
    _validate(instance, fschema_opt, restrictive=True, removal=True, defaults=False, rank_0=True)

    # Ensure the output filename does not end with .yaml or .yml
    if foutput.endswith(".yaml"):
        foutput = foutput[:-5]
    elif foutput.endswith(".yml"):
        foutput = foutput[:-4]

    sfx_str = "-analysis.yaml"
    write_yaml(instance, foutput + sfx_str)
    return foutput + sfx_str

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
