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
        if e.validator == "required":
            for k in e.validator_value:
                if k not in e.instance.keys():
                    mypath = e.absolute_path.copy()
                    mypath.append(k)
                    v = nested_get(defaults, mypath)
                    if isinstance(v, dict) or isinstance(v, list) or v in ["name", "material"]:
                        # Too complicated to just copy over default, so give it back to the user
                        raise (e)
                    else:
                        print("WARNING: Missing value,", list(mypath), ", so setting to:", v)
                        nested_set(instance, mypath, v)
        else:
            raise (e)
    return instance


def simple_types(indict):
    rv = indict
    for k in rv.keys():
        if type(rv[k]) == type(np.array([])):
            rv[k] = rv[k].tolist()
        elif isinstance(rv[k], (float, int, dict, bool, str)):
            pass
        # simple_types iterates through dictionaries, but skips lists and tuples, which may have numpy objects within
        elif isinstance(rv[k], (list, tuple)):
            for l in rv[k]:
                try:
                    simple_types(l)
                except:
                    # Will error out if there its already a simple type and there are no keys
                    continue
        else:
            rv[k] = ""

        try:
            simple_types(rv[k])
        except Exception:
            continue
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
    schema_dict = fschema if isinstance(fschema, dict) else load_yaml(fschema)
    input_dict = finput if isinstance(finput, dict) else load_yaml(finput)
    if defaults:
        DefaultValidatingDraft7Validator(schema_dict).validate(input_dict)
    else:
        json.Draft7Validator(schema_dict).validate(input_dict)
    return input_dict


# ---------------------
def load_geometry_yaml(finput):
    return _validate(finput, fschema_geom)


def load_modeling_yaml(finput):
    return _validate(finput, fschema_model)


def load_analysis_yaml(finput):
    return _validate(finput, fschema_opt)


def write_geometry_yaml(instance, foutput):
    _validate(instance, fschema_geom, defaults=False)
    sfx_str = ".yaml"
    if foutput[-5:] == sfx_str:
        sfx_str = ""
    write_yaml(instance, foutput + sfx_str)


def write_modeling_yaml(instance, foutput):
    _validate(instance, fschema_model, defaults=False)
    sfx_str = ".yaml"
    if foutput[-5:] == sfx_str:
        foutput = foutput[-5:]
    elif foutput[-4:] == ".yml":
        foutput = foutput[-4:]
    sfx_str = "-modeling.yaml"

    instance2 = simple_types(instance)
    write_yaml(instance2, foutput + sfx_str)


def write_analysis_yaml(instance, foutput):
    _validate(instance, fschema_opt, defaults=False)
    sfx_str = ".yaml"
    if foutput[-5:] == sfx_str:
        foutput = foutput[-5:]
    elif foutput[-4:] == ".yml":
        foutput = foutput[-4:]
    sfx_str = "-analysis.yaml"
    write_yaml(instance, foutput + sfx_str)

def remove_numpy(fst_vt):
    # recursively move through nested dictionary, remove numpy data types
    # for formatting dictionaries before writing to yaml files

    def get_dict(vartree, branch):
        return reduce(operator.getitem, branch, vartree)

    def loop_dict(vartree, branch):
        if not isinstance(vartree, dict):
            return fst_vt
        for var in vartree.keys():
            branch_i = copy.copy(branch)
            branch_i.append(var)
            if isinstance(vartree[var], dict):
                loop_dict(vartree[var], branch_i)
            else:
                data_type = type(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])

                if data_type in [np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = int(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])
                elif data_type in [np.single, np.double, np.longdouble, np.csingle, np.cdouble, np.float_, np.float16, np.float32, np.float64, np.complex64, np.complex128]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = float(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])
                elif data_type in [np.bool_]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = bool(get_dict(fst_vt, branch_i[:-1])[branch_i[-1]])
                elif data_type in [np.ndarray]:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = get_dict(fst_vt, branch_i[:-1])[branch_i[-1]].tolist()
                elif data_type in [list,tuple]:
                    for item in get_dict(fst_vt, branch_i[:-1])[branch_i[-1]]:
                        remove_numpy(item)

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
