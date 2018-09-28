from numpy import matrix
from random import randint
from math import floor
from contextlib import contextmanager
# import pdb


def __is_positive_seq(seq):
    for e in seq:
        if e is None or e < 0:
            return False
    return True


def __validate_xcol(cols):
    if cols is None:
        raise ValueError('Input must be a scalar or array!')
    elif cols == [] or len(cols) < 2:
        raise ValueError('There must have at least two elements!')
    elif isinstance(cols, list) and not __is_positive_seq(cols):
        raise ValueError('Input must have positive indexes only SEQ!')


@contextmanager
def rsfile(fname, stg='tr', sys='arx'):
    ''' Create a file into results folder with the pattern:

                sys_fname_stage.rs
    '''
    fileline = '{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{}\n'
    header = ['stdev', 'aic', 'fpe', 'order', 'delay', 'params']
    name = '_'.join([sys, fname, stg])
    fullname = 'results/' + name + '.rs'
    file = open(fullname, 'w')
    file.write(fileline.format(*header))
    try:
        yield file
    finally:
        file.close()


@contextmanager
def open_matrix(fname, sep='\t'):
    ''' Read a given file and create a matrix with its contents

    Parameters
    ----------
    fname : str
        A file name in examples folder
    sep: str, defaults to '\t'
        The file separator

    Returns
    -------
    dots : unmpy matrix
        A matrix with the file data
    '''
    file = open(fname, mode='r')
    try:
        dots = []
        fcontent = file.readlines()
        for i in range(len(fcontent)):
            fcontent[i] = fcontent[i].strip('\n\t ')
            fcontent[i] = fcontent[i].strip(' ').split(sep)
            fcontent[i] = [float(num) for num in fcontent[i]]
            dots.append(fcontent[i][:])
        yield matrix(dots)
    finally:
        file.close()


def r_dots(fname, columns=[0, 1], sep=' '):
    ''' Read a file separated by spaces into two matrices.

    Paramaters
    ----------
    fname : str
        The name of a file inside 'examples' folder
    columns : list of int
        Columns indexes, output is the last element input are the others
    sep : str, defaults to ' '
        The file seperator character

    Returns
    -------
    u : np matrix (n, M)
        The i-column, where i is in xcol
    y : np matrix (n, 1)
        The last column of the file
    '''
    __validate_xcol(columns)
    ninps = len(columns) - 1
    data = open('examples/' + fname, 'r')
    data = data.readlines()
    dots = []
    for i in range(len(data)):
        data[i] = data[i].strip('\n').strip(' ').strip(sep).split(sep)
        row = []
        for elm in data[i]:
            if elm not in ['', '\n', '\t']:
                row.append(float(elm))
        dots.append([row[x] for x in columns])
    dots = matrix(dots)
    return dots[:, :ninps], dots[:, -1]


def separate_subset(fname, tsize=.3):
    '''
    Parameters
    ----------
    fname : str
        A file inside examples folder
    tsize : float
        The fraction of the set to be used as validation
    '''
    data = open('examples/' + fname, 'r')
    data = data.readlines()
    n_vpoints = int(floor(len(data) * tsize))
    n_tpoints = len(data) - n_vpoints
    vpoints = []
    while len(vpoints) < n_vpoints:
        p = randint(0, len(data) - 1)
        if p not in vpoints:
            vpoints.append(p)
    vpoints.sort()

    fprefix = fname.split('.')[0]
    validation = open('examples/validation/' + fprefix + '_val.dat', 'w')
    train = open('examples/training/' + fprefix + '_train.dat', 'w')
    k = 0
    for i in range(n_vpoints + n_tpoints):
        if k < n_vpoints and i == vpoints[k]:
            validation.write(data[i])
            k += 1
        else:
            train.write(data[i])

    validation.close()
    train.close()
