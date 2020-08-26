"""
John Jasa
May 2020

This script parses a file containing OpenMDAO components and produces
docstrings for those components. It makes some assumptions about the
format of the code. Specifically, it reads the `add_input` and `add_output`
statements from the components and generates docstrings based on that.

Example usage:
python convert_docstrings.py tower.py

This creates a file called `tower_docstrings.py` which contains the
modified components. These new components can then be used as-is or copied
and pasted into the older file.
"""

import sys
import textwrap


filename = sys.argv[1]
with open(filename) as f:
    content = f.readlines()
orig_content = content.copy()
orig_content = [x.strip('\n') for x in orig_content]
content = [x.strip() for x in content]

doc_lines = []

in_class_flag = False
for line in content:
    
    if len(line) > 0:
        if '#' in line[0]:
            continue

    if 'Component):' in line:
        class_name = line.split()[1]
        in_class_flag = True
        
        parsed_data = {}
        parsed_data['inputs'] = {}
        parsed_data['outputs'] = {}
        new_add_statements = []
        
    elif '.add_input(' in line or '.add_output(' in line or '.add_discrete_input(' in line or '.add_discrete_output(' in line:
        if in_class_flag:
            
            if '_input(' in line:
                is_input = True
            else:
                is_input = False
                
            if '_discrete_' in line:
                discrete_string = 'discrete_'
            else:
                discrete_string = ''
            
            # Split statements on commas to roughly group the terms
            comma_sep_line = line.split(',')
            
            # Strip whitespace from each entry
            stripped_line = [x.strip() for x in comma_sep_line]
            
            # Need this logic to handle if the name used '' or ""
            if is_input:
                var_name = stripped_line[0].split(f'add_{discrete_string}input(')[1][1:-1]
            else:
                var_name = stripped_line[0].split(f'add_{discrete_string}output(')[1][1:-1]
            
            # Get the value type from the line
            val_str = stripped_line[1]
            
            # Some logic to handle numpy arrays
            left = val_str.count('(')
            right = val_str.count(')')
            i_paren = 1
            while left != right:
                val_str += ', ' + stripped_line[1+i_paren]
                i_paren += 1
                right = val_str.count(')')
            if '=' in val_str:
                val_str = val_str.split('=')[1].strip()
                
            if 'np' in val_str:
                split_str = val_str.split('(')[-1].split(')')[0]
                type_ = f'numpy array[{split_str}]'
                type_ = type_.replace('[[', '[')
                type_ = type_.replace(']]', ']')                
            else:
                try:
                    float(val_str)
                    if 'discrete' in discrete_string:
                        type_ = 'TODO: add type by hand, could not be parsed automatically'
                    else:
                        type_ = 'float'
                        
                except:
                    type_ = 'TODO: add type by hand, could not be parsed automatically'
                    
            if 'desc' in line:
                desc = line.split('desc')[1]
                if '\'' in desc:
                    desc = desc.split('\'')[1]
                elif '\"' in desc:
                    desc = desc.split('\"')[1]
                    
                desc = ' '.join([x.strip() for x in desc.split()])
                if len(desc) > 80:
                    desc = "\n        ".join(textwrap.wrap(desc, 80))
                    
                split_line = line.split(',')
                new_split_line = []
                for i_string, substring in enumerate(split_line):
                    if 'desc' in substring:
                        if i_string == len(split_line) - 1:
                            desc_last = True
                        else:
                            desc_last = False
                    else:
                        new_split_line.append(substring)
                        
                # Need special logic is desc is the last entry in the add_ call
                # or if units comes after
                if desc_last:
                    new_add_statement = line.split('desc')[0].strip()[:-1] + ')'
                else:
                    new_add_statement = ', '.join(new_split_line)
                    
                new_add_statements.append(new_add_statement) 
                
            else:
                desc = ''
                new_add_statements.append(line)
                
            # If units are included, parse the line to get the actual units
            if 'units' in line:
                
                # Split the line on the word units; grab everything after that
                post_units = line.split('units')[1]
                
                # Remove all whitespace from the string
                post_units = ''.join(post_units.split())
                
                # Split the line on ',' or ')' to remove extra formatting
                post_units = post_units.split(',')[0]
                post_units = post_units.split(')')[0]
                
                # Get only the actual units, no quotation marks or equals signs
                units = post_units[2:-1]
                
                # Add formatting to make it look good on the docstring
                units = f', [{units}]'
                
            # If no units in line, add placeholder string
            else:
                units = ''
                
            if is_input:
                parsed_data['inputs'][var_name] = {'type_' : type_, 'desc' : desc, 'units' : units}
            else:
                parsed_data['outputs'][var_name] = {'type_' : type_, 'desc' : desc, 'units' : units}
                
    elif 'compute(' in line:
        in_class_flag = False
        
        doc_lines.append(class_name)
        doc_lines.append('\"\"\"')
        doc_lines.append('TODO : replace this with docstring')
        doc_lines.append('')
        
        doc_lines.append('Parameters')
        doc_lines.append('----------')
        for var_name in parsed_data['inputs']:
            data = parsed_data['inputs'][var_name]
            doc_lines.append(f"{var_name} : {data['type_']}{data['units']}")
            doc_lines.append(f"    {data['desc']}")
            
        doc_lines.append('')
        doc_lines.append('Returns')
        doc_lines.append('-------')
        for var_name in parsed_data['outputs']:
            data = parsed_data['outputs'][var_name]
            doc_lines.append(f"{var_name} : {data['type_']}{data['units']}")
            doc_lines.append(f"    {data['desc']}")
        doc_lines.append('')
        doc_lines.append('\"\"\"')
        
        doc_lines.append('')
        doc_lines.append('')
        output_flag = False
        for add_line in new_add_statements:
            if '_output' in add_line and not output_flag:
                doc_lines.append('')
                output_flag = True
            stripped_line = ', '.join([x.strip() for x in add_line.split(',')])
            doc_lines.append(stripped_line)
            
        doc_lines.append('')
        doc_lines.append('')
        
for i_line, line in enumerate(doc_lines):
    doc_lines[i_line] = f'    {line}'
        
new_filename = filename.split('.')[0] + '_docstrings.' + filename.split('.')[1]

full_file = []
in_class_flag = False
i_docs = 0
for line in orig_content:
    
    if len(line) > 0:
        if '#' in line[0]:
            full_file.append(line)
            continue

    if 'Component):' in line:
        class_name = line.split()[1]
        in_class_flag = True
        full_file.append(line)
        adds_done = False
        
        smaller_doc_lines = doc_lines[i_docs:]
        in_docstring = False
        for new_line in smaller_doc_lines:
            if '\"\"\"' in new_line:
                if not in_docstring:
                    in_docstring = True
                else:
                    in_docstring = False
                    full_file.append(new_line)
                    i_docs += 2
                    break
            if in_docstring:
                full_file.append(new_line)
                i_docs += 1
                
    elif 'self.add_' in line and in_class_flag and not adds_done:
        smaller_doc_lines = doc_lines[i_docs:]
        in_adds = False
        first_output = True
        for new_line in smaller_doc_lines:
            if 'self.add_' in new_line:
                
                if '_output' in new_line and first_output:
                    full_file.append('')
                    first_output = False
                    
                full_file.append(f'    {new_line}')
                i_docs += 1
            elif 'Component)' in new_line:
                i_docs += 1
                adds_done = True
                break
            else:
                if len(new_line.strip()) > 0:
                    full_file.append(f'    {new_line}')
                i_docs += 1
        
    elif 'compute(' in line:
        in_class_flag = False
        full_file.append(line)
        
    elif 'self.add_' in line and 'add_subsystem' not in line:
        pass
                
    else:
        full_file.append(line)

with open(new_filename, 'w') as f:
    for line in full_file:
        f.write(line + '\n')

        