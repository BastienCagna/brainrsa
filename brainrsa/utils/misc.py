import numpy as np
import nibabel as nb
from joblib import cpu_count
from nilearn.image import new_img_like


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


def check_mask(mask_f, out_f=None, threshold=0.0):
    mask = nb.load(mask_f) if isinstance(mask_f, str) else mask_f
    dt = np.array(mask.dataobj)
    if len(np.unique(dt)) > 2:
        dt[dt <= threshold] = 0
        dt[dt > threshold] = 1
    mask = new_img_like(mask, dt)
    if out_f:
        nb.save(mask, out_f)
    return mask


class GroupIterator(object):
    """Group iterator

    Provides group of features for search_light loop
    that may be used with Parallel.

    Parameters
    ----------
    n_features : int
        Total number of features

    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'. Defaut is 1
    """
    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        for list_i in split:
            yield list_i

