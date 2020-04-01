import numpy as np


################# UTILS ########################################################
#def argmax(arr):
#    return np.unravel_index(np.argmax(arr, axis=None), arr.shape)


def tri_num(n):
    """ Compute triangular number """
    if type(n) != int or n <= 0:
        raise ValueError("n must be a positive integer.")
    return int(n * (n+1) / 2)


def root_tri_num(t):
    """ Reverse tri_num """
    if type(t) != int or t <= 0:
        raise ValueError("t must be a positive integer.")
    r = (-1 + np.sqrt(1 + 8 * t)) / 2 # + 1
    if np.modf(r)[0] != 0:
        raise ValueError("t is not a triangular number.")
    return int(r)


#def moving_average(a, n) :
#    ret = np.cumsum(a, dtype=float)
#    ret[n:] = ret[n:] - ret[:-n]
#    return ret[n - 1:] / n


#def mult_replace(in_str, **kwargs):
#    for key, val in kwargs.items():
#        in_str = in_str.replace("[" + k + "]", val)
#    return in_str


