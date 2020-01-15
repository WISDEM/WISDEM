import os, itertools
import numpy as np

def save_case_matrix_direct(case_list, dir_matrix):
    ### assumes all elements of the list are dict for that case that has the same keys!
    if not os.path.exists(dir_matrix):
            os.makedirs(dir_matrix)
    ofh = open(os.path.join(dir_matrix,'case_matrix.txt'),'w')
    case = case_list[0]
    for key in case.keys():
        k = key[0]
        ofh.write("%s  " % k)
    ofh.write("\n")
    for key in case.keys():
        k = key[1]
        ofh.write("%s  " % k)
    ofh.write("\n")
    for i in range(len(case_list)):
        case = case_list[i]
        for key in case.keys():
            ofh.write(str(case[key]))
            ofh.write("  ")
        ofh.write("\n")
    ofh.close()


def save_case_matrix(matrix_out, change_vars, dir_matrix):
    # save matrix file
    if type(change_vars[0]) is tuple:
        n_header_lines = len(change_vars[0])
    else:
        change_vars = [(var,) for var in change_vars]
        n_header_lines = 1

    n_cases = np.shape(matrix_out)[0]
    matrix_out = np.hstack((np.asarray([[i] for i in range(n_cases)]), matrix_out))

    change_vars = [('Case_ID',)+('',)*(n_header_lines-1)] + change_vars
    # col_len = [max([len(val) for val in matrix_out[:,j]] + [len(change_vars[j][0]), len(change_vars[j][1])]) for j in range(len(change_vars))]
    col_len = [max([len(str(val)) for val in matrix_out[:,j]] + [len(change_vars[j][header_i]) for header_i in range(n_header_lines)]) for j in range(len(change_vars))]

    text_out = []
    for header_i in range(n_header_lines):
        text_out.append(''.join([val.center(col+2) for val, col in zip([var[header_i] for var in change_vars], col_len)])+'\n')

    for row in matrix_out:
        row_str = ''
        for val, col in zip(row, col_len):
            if val is not str:
                val = str(val)
            row_str += val.center(col+2)
        row_str += '\n'
        text_out.append(row_str)

    if not os.path.exists(dir_matrix):
            os.makedirs(dir_matrix)
    ofh = open(os.path.join(dir_matrix,'case_matrix.txt'),'w')
    for row in text_out:
        ofh.write(row)
    ofh.close()

def case_naming(n_cases, namebase=None):
    # case naming
    case_name = [('%d'%i).zfill(len('%d'%(n_cases-1))) for i in range(n_cases)]
    if namebase:
        case_name = [namebase+'_'+caseid for caseid in case_name]

    return case_name

def convert_str(val):
    def try_type(val, data_type):
        try:
            data_type(val)
            return True
        except:
            return False
#        return isinstance(val, data_type)  ### this doesn't work b/c of numpy data types; they're not instances of base types
    def try_list(val):
        try:
            val[0]
            return True
        except:
            return False

    if try_type(val, int) and int(val) == float(val):
        return int(val)
    elif try_type(val, float):
        return float(val)
    elif val=='True':
        return True
    elif val=='False':
        return False
    # elif type(val)!=str and try_list(val):
    #     return ", ".join(['{:}'.format(i) for i in val])
    else:
        return val

def CaseGen_General(case_inputs, dir_matrix='', namebase='', save_matrix=True):
    """ Cartesian product to enumerate over all combinations of set of variables that are changed together"""

    # put case dict into lists
    change_vars = sorted(case_inputs.keys())
    change_vals = [case_inputs[var]['vals'] for var in change_vars]
    change_group = [case_inputs[var]['group'] for var in change_vars]

    # find number of groups and length of groups
    group_set = list(set(change_group))
    group_len = [len(change_vals[change_group.index(i)]) for i in group_set]

    # case matrix, as indices
    group_idx = [range(n) for n in group_len]
    matrix_idx = list(itertools.product(*group_idx))

    # index of each group
    matrix_group_idx = [np.where([group_i == group_j for group_j in change_group])[0].tolist() for group_i in group_set]

    # build final matrix of variable values
    matrix_out = []
    for i, row in enumerate(matrix_idx):
        row_out = [None]*len(change_vars)
        for j, val in enumerate(row):
            for g in matrix_group_idx[j]:
                row_out[g] = change_vals[g][val]
        matrix_out.append(row_out)
    matrix_out = np.asarray(matrix_out)
    n_cases = np.shape(matrix_out)[0]

    # Save case matrix
    if save_matrix:
        if not dir_matrix:
            dir_matrix = os.getcwd()
        save_case_matrix(matrix_out, change_vars, dir_matrix)

    case_list = []
    for i in range(n_cases):
        case_list_i = {}
        for j, var in enumerate(change_vars):
            case_list_i[var] = convert_str(matrix_out[i,j])
        case_list.append(case_list_i)

    # case naming
    case_name = case_naming(n_cases, namebase=namebase)

    return case_list, case_name


if __name__ == "__main__":

    case_inputs = {}
    case_inputs[("Fst","TMax")] = {'vals':[10.], 'group':0}
    case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}

    case_inputs[("InflowWind","HWindSpeed")] = {'vals':[8., 9., 10., 11., 12.], 'group':1}
    case_inputs[("ElastoDyn","RotSpeed")] = {'vals':[9.156, 10.296, 11.431, 11.89, 12.1], 'group':1}
    case_inputs[("ElastoDyn","BlPitch1")] = {'vals':[0., 0., 0., 0., 3.823], 'group':1}
    case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
    case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]

    case_inputs[("ElastoDyn","GenDOF")] = {'vals':['True','False'], 'group':2}

    case_list, case_name = CaseGen_General(case_inputs, 'C:/Users/egaertne/WISDEM/AeroelasticSE/src/AeroelasticSE/', 'testing')
