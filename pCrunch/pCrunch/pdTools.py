'''
Some tools to ease management of batch analysis data in pandas
'''
import pandas as pd
from pCrunch import Processing

def dict2df(sumstats, names=None):
    '''
    Build pandas datafrom from list of summary statistics

    Inputs:
    -------
    sumstats: list
        List of the dictionaries loaded from post_process.load_yaml
    names: list, optional
        List of names for each run. len(sumstats)=len(names)

    Returns:
    --------
    df: pd.DataFrame
        pandas dataframe 
    '''
    if isinstance(sumstats, list):
        if not names:
            names = ['dataset_' + str(i) for i in range(len(sumstats))]
        data_dict = {(name, outerKey, innerKey): values 
                        for name, sumdata in zip(names, sumstats)
                        for outerKey, innerDict in sumdata.items() 
                        for innerKey, values in innerDict.items()}
    else:
        data_dict = {(outerKey, innerKey): values 
                        for outerKey, innerDict in sumstats.items()
                        for innerKey, values in innerDict.items()}

    # Make dataframe
    df = pd.DataFrame(data_dict)

    return df


def df2dict(df):
    '''
    Build dictionary from pandas dataframe

    Parameters:
    -----------
    df: DataFrame
        DataFrame with summary stats (probably). Cannot have len(multiindex) > 3

    Returns:
    --------
    dfd: dict
        dictionary containing re-structured dataframe inputs
    '''
    if len(df.columns[0]) == 3:
        dfd = [{level: df.xs(dset, axis=1).xs(level, axis=1).to_dict('list')
              for level in df.columns.levels[1]} 
              for dset in df.columns.levels[0]]
    elif len(df.columns[0]) == 2:
        dfd = {level: df.xs(dset, axis=1).xs(level, axis=1).to_dict('list')
              for level in df.columns.levels[0]} 
    elif len(df.columns[0]) == 1:
        dfd = df.to_dict('list')
    else:
        raise TypeError('Converting DataFrames with multiindex > 3 to dictionaries is not supported')

    return dfd

def yaml2df(filename, names=[]):
    '''
    Read yaml containing summary stats into dataframe

    Parameters:
    -------
    filename:
        Name of yaml file to load. 

    '''

    data_dict = Processing.load_yaml('test.yaml', package=0)

    level = data_dict
    li = 0 # level index
    while isinstance(level, dict):
        li = li + 1
        if isinstance(level, dict):
            level = level[list(level.keys())[0]]

    data_list = []
    if li == 3:
        for key in data_dict.keys():
            data_list.append(data_dict[key])
        if not names:
            names = ['dataset_' + str(i) for i in range(len(data_list))]

    elif li == 2:
        data_list.append(data_dict)
        names = []
    else:
        raise TypeError('{} is improperly structured for yaml2df.'.format(filename))

    df = dict2df(data_list, names=names)

    return df
