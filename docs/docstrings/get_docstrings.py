# -*- coding: utf-8 -*-
"""
Parse Python source code and get or print docstrings.
https://gist.github.com/SpotlightKid/1548cb6c97f2a844f72d
"""

__all__ = ("get_docstrings", "print_docstrings")

import ast
import os.path as osp
from itertools import groupby
from pathlib import Path
from docstring_parser import parse

NODE_TYPES = {ast.ClassDef: "Class", ast.FunctionDef: "Function/Method", ast.Module: "Module"}


def get_docstrings(source):
    """Parse Python source code and yield a tuple of ast node instance, name,
    line number and docstring for each function/method, class and module.

    The line number refers to the first line of the docstring. If there is
    no docstring, it gives the first line of the class, funcion or method
    block, and docstring is None.

    """
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, tuple(NODE_TYPES)):
            docstring = ast.get_docstring(node)
            lineno = getattr(node, "lineno", None)

            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                # lineno attribute of docstring node is where string ends
                lineno = node.body[0].lineno - len(node.body[0].value.s.splitlines()) + 1

            yield (node, getattr(node, "name", None), lineno, docstring)


def print_docstrings(source, module="<string>", print_flag=True):
    """Parse Python source code from file or string and print docstrings.

    For each class, method or function and the module, prints a heading with
    the type, name and line number and then the docstring with normalized
    indentation.

    The module name is determined from the filename, or, if the source is passed
    as a string, from the optional `module` argument.

    The line number refers to the first line of the docstring, if present,
    or the first line of the class, funcion or method block, if there is none.

    Output is ordered by type first, then name.

    """
    if hasattr(source, "read"):
        filename = getattr(source, "name", module)
        module = osp.splitext(osp.basename(filename))[0]
        source = source.read()

    docstrings = sorted(get_docstrings(source), key=lambda x: (NODE_TYPES.get(type(x[0])), x[1]))
    grouped = groupby(docstrings, key=lambda x: NODE_TYPES.get(type(x[0])))

    if print_flag:
        for type_, group in grouped:
            for node, name, lineno, docstring in group:
                name = name if name else module
                heading = "%s '%s', line %s" % (type_, name, lineno or "?")
                print(heading)
                print("-" * len(heading))
                print("")
                print(docstring or "")
                print("\n")

    return grouped


def get_all_docstrings():
    source_dir = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
    allpy = Path(source_dir).rglob("*.py") if osp.exists(source_dir) else []

    parsed_args = [["File", "Class", "Var Name", "Var Type", "Var Units", "Var Desc"]]
    parsed_dict = {}

    for f in allpy:
        if str(f).find("__init__") >= 0:
            continue
        froot = str(f).replace(str(source_dir), "")

        try:
            with open(f) as fp:
                fgroup = print_docstrings(fp, print_flag=False)
        except Exception:
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
                    if type_name is not None:
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

    return parsed_dict

if __name__ == "__main__":
    import sys

    with open(sys.argv[1]) as fp:
        print_docstrings(fp)
