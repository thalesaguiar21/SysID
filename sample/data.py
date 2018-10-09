from numpy import matrix
from random import randint
from math import floor
from contextlib import contextmanager
from sample.identification import Structure, Stage


@contextmanager
def rsfile(fname, stg=Stage.TRAINING, struc=Structure.ARX):
    """ Create a file into results folder with the pattern:
    struc_fname_stage.rs

    Args:
        fname (str): the file path
        stg (Stage): the identification stage. Defaults to TRAINING
        struc (Structure): the system structure. Defaults to ARX

    Yields:
        file: The file to write the results

    Raises:
        ValueError: if struc is not supported
        ValueError: if stg is not supported
    """
    if isinstance(struc, Structure):
        raise ValueError('Unknown structure: ' + struc)
    if isinstance(stg, Stage):
        raise ValueError('Unknown stage: ' + stg)

    fileline = '{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{}\n'
    header = ['stdev', 'aic', 'fpe', 'order', 'delay', 'params']
    name = '_'.join([struc, fname, stg])
    fullname = 'results/' + name + '.rs'
    file = open(fullname, 'w')
    file.write(fileline.format(*header))
    try:
        yield file
    finally:
        file.close()


@contextmanager
def open_matrix(fname, sep='\t'):
    """ Read a given file and create a matrix with its contents

    Args:
        fname (str): A file name in examples folder
        sep (str): The file separator. Defaults to '\t'

    Returns:
        A matrix with the file data
    """
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


def separate_subset(fname, vsize=.3):
    """ Randomly separate a data file into train/test subsets without intersect
    ion.

    Args:
        fname (str): A file inside examples folder.
        vsize (float): The fraction of the set to be used as validation.

    Returns:
        Create two files into training and validation folders under example.
    """
    data = open('examples/' + fname, 'r')
    data = data.readlines()
    n_vpoints = int(floor(len(data) * vsize))
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
