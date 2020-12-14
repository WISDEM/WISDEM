import os
import csv
from pathlib import Path

import wisdem as wstart
import getdocstrings
from docstring_parser import parse

source_dir = os.path.dirname(os.path.realpath(wstart.__file__))
allpy = Path(source_dir).rglob("*.py") if os.path.exists(source_dir) else []

parsed_args = [["File", "Class", "Var Name", "Var Type", "Var Units", "Var Desc"]]
parsed_dict = {}

for f in allpy:
    if str(f).find("__init__") >= 0:
        continue
    froot = str(f).replace(str(source_dir), "")

    try:
        with open(f) as fp:
            fgroup = getdocstrings.print_docstrings(fp, print_flag=False)
    except:
        print("Could not parse", str(f))
        continue

    for type_, group in fgroup:
        if type_ != "Class":
            continue

        for node, name, lineno, docstring in group:
            parsed_docs = parse(docstring)
            if not hasattr(parsed_docs, "params"):
                continue
            # print(str(f),len(parsed_docs.params))

            for k in range(len(parsed_docs.params)):
                arg_name = parsed_docs.params[k].arg_name
                arg_description = parsed_docs.params[k].description

                type_name = parsed_docs.params[k].type_name
                if not type_name is None:
                    tok = type_name.split(",")
                    arg_type = tok[0]
                    isize = arg_type.find("[")
                    if isize > 0:
                        arg_type = arg_type[:isize]
                else:
                    arg_type = ""

                arg_units = "" if len(tok) <= 1 else tok[1]
                ibracket0 = arg_units.find("[")
                ibracket1 = arg_units.find("]")
                if ibracket0 >= 0 and ibracket1 >= 0:
                    arg_units = arg_units[(ibracket0 + 1) : ibracket1]
                # print('args',parsed_docs.params[k].args)
                # print('default',parsed_docs.params[k].default)
                # print('is_optional',parsed_docs.params[k].is_optional)
                newentry = [froot, name, arg_name, arg_type, arg_units, arg_description]
                parsed_args.append(newentry)
                parsed_dict[arg_name] = arg_description

# Write out these parsed args for later
# with open('wisdem_docargs.csv','w') as fw:
#    writer = csv.writer(fw)
#    writer.writerows(parsed_args)

# Now read in dumps from WISDEM
wisdem_outputs = ["iea15mw_output.csv", "nrel5mw_output.csv"]
master_guide = [["Variable", "Units", "Description"]]
temp_guide = []
varlist = []
for infile in wisdem_outputs:
    with open(infile, "r") as fread:
        ilist = list(csv.reader(fread, dialect="excel"))
        temp_guide.extend(ilist)

# Crude way of making unique vars
temp_guide.sort(key=lambda r: r[0])
for k in temp_guide:
    if k[0] in varlist:
        continue
    varlist.append(k[0])
    master_guide.append([k[0], k[1], k[3]])

# Now fold in the doc strings
for k in master_guide:
    var_name = k[0].split(".")[-1]
    if var_name in parsed_dict:
        idesc = parsed_dict[var_name]
        k[2] += "" if idesc is None else idesc

# Write out these parsed args for later
with open("variable_guide.csv", "w") as fw:
    writer = csv.writer(fw)
    writer.writerows(master_guide)
