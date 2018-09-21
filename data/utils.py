from numpy import matrix
from random import randint
from math import floor


def __is_positive_seq(seq):
    for e in seq:
        if e < 0:
            return False
    return True


def __validate_xcol(xcol):
    if xcol is None:
        raise ValueError('Input must be a scalar or array!')
    elif xcol == []:
        raise ValueError('Input must have at least one element!')
    elif isinstance(xcol, list) and not __is_positive_seq(xcol):
        raise ValueError('Input must have positive indexes only SEQ!')
    elif not isinstance(xcol, list) and xcol < 0:
        raise ValueError('Input must have positive indexes only!')


def __validate_ycol(ycol):
    if ycol is None:
        raise ValueError('Output must be a scalar.')
    if ycol < 0:
        raise ValueError('Output must be positive.')


def __get_index(inp, out):
    idx = []
    ninps = 1
    if isinstance(inp, list):
        idx = inp[:]
        ninps = len(inp)
    else:
        idx.append(inp)
    idx.append(out)
    return idx, ninps


def r_dots(fname, xcol=0, ycol=-1, sep=' '):
    ''' Read a file separated by spaces into two matrices.

    Paramaters
    ----------
    fname : str
        The name of a file inside 'examples' folder
    xcol : int, defaults to 0
        The column of x
    ycol : int, defaults to -1
        The column of y
    sep : str, defaults to ' '
        The file seperator character

    Returns
    -------
    u : np matrix (n, M)
        The i-column, where i is in xcol
    y : np matrix (n, 1)
        The last column of the file
    '''
    __validate_xcol(xcol)
    __validate_ycol(ycol)
    indexes, ninps = __get_index(xcol, ycol)
    data = open('examples/' + fname, 'r')
    data = data.readlines()
    dots = []
    for i in xrange(len(data)):
        data[i] = data[i].strip('\n').strip(' ').strip(sep).split(sep)
        row = []
        for elm in data[i]:
            if elm not in ['', '\n', '\t']:
                row.append(float(elm))
        dots.append([row[x] for x in indexes])
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
