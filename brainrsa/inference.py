import numpy as np

from sklearn.externals.joblib import Parallel, delayed, cpu_count

from scipy.spatial.distance import squareform
from scipy.stats.mstats import spearmanr, rankdata
import scipy.spatial.distance as dist
from scipy.stats import ttest_rel

import matplotlib.pyplot as plt

import time


def nperms_indexes(nobj, nperm):
    """
    Generates permutations

    Perms contains the permutations of a square distance matrix
    of size nobj generated by permuting simultaneously the nobj columns and
    rows.
    Data is assumed to be of size [npair,ncol] where npair = nobj*(nobj-1)/2,
    and distances within each column are sorted as expected by squareform().

    Parameters
    ===========
    nobj: unsigned int
        Number of objects (= number of rows = number of cols) of the RDM.

    nperm: unsigned int
        Number of permutations

    Returns
    ========
    perms: npairs x nperms numpy array
        Permuted elements indexes

    Credits
    ========
    Inspired by Matlab script of:
    Bruno L. Giordano, February 2017
    Institute of Neuroscience and Psychology, University of Glasgow
    brungio@gmail.com

    """
    # tmpperms = list of index permutation
    r = np.random.randn(nperm, nobj)
    tmpperms = np.argsort(r)

    # Number of pair of elements (triangular number)
    npairs = tri_num(nobj - 1)
    # Create a matrix with the index of each element
    idx = dist.squareform(np.array(range(npairs)))

    # Initialize permutation
    perms = np.zeros((nperm, npairs), dtype=int)
    for wichperm in range(nperm):
        # Generate the correct perumutation (taking care columns and rows
        # dependencies)
        tmp = tmpperms[wichperm, :]
        iidx = idx[np.tile(tmp, nobj), np.repeat(tmp, nobj)].reshape(nobj, nobj)
        # Transform matrix into vector
        perms[wichperm, :] = dist.squareform(iidx)

    return perms


def _group_iter_compare_rdms(brain_v, model_vectors, distance, perms,
                             seed=0, n_seeds=0, verbose=0, start_time=None):
    """ Mesure distance between brain RDM and several model RDMs """

    probas = np.zeros((len(model_vectors),), dtype=float)
    for im in range(model_vectors.shape[0]):
        true_score = _cross_vect_score(brain_v, model_vectors[im], distance)

        random_scores = np.zeros((len(perms),), dtype=float)
        for p, perm in enumerate(perms):
            perm_v = model_vectors[im][perm]
            random_scores[p] = _cross_vect_score(brain_v, perm_v, distance)

        random_scores = np.array(random_scores)

        probas[im] = (np.sum(random_scores > true_score) + 1) / len(perms)

    if verbose > 0 and n_seeds > 0:
        if seed % np.floor(n_seeds / 100) == 0:
            if start_time:
                per_seed = (time.time() - start_time) / (seed + 1)
                remain = (n_seeds - seed) * per_seed
                hours, rem = divmod(remain, 3600)
                minutes, seconds = divmod(rem, 60)
                remaining = "remaining {:0>2}:{:0>2}:{:05.2f} ({:05.3f}/seed)".\
                    format(int(hours), int(minutes), seconds, per_seed)
            else:
                remaining = ""

            print('seed {} / {}: {}\t{}'.format(seed, n_seeds, probas, remaining))
    return probas


def _group_iter_numerical_comparison(brain_v, masks, perms, seed=0, n_seeds=0,
                                     verbose=0, start_time=None):
    probas = np.empty((len(masks),))
    for im, mask in enumerate(masks):
        print(len(brain_v[mask==1]), len(brain_v[mask==0]))
        _, p = ttest_rel(brain_v[mask][:380], brain_v[mask == 0], nan_policy='omit')
        probas[im] = p

    print(p)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(brain_v[mask==1])
    plt.subplot(2, 1, 2)
    plt.hist(brain_v[mask == 0])
    plt.show()

    if verbose > 0 and n_seeds > 0:
        if seed % np.floor(n_seeds / 100) == 0:
            if start_time:
                per_seed = (time.time() - start_time) / (seed + 1)
                remain = (n_seeds - seed) * per_seed
                hours, rem = divmod(remain, 3600)
                minutes, seconds = divmod(rem, 60)
                remaining = "remaining {:0>2}:{:0>2}:{:05.2f} ({:05.3f}/seed)".\
                    format(int(hours), int(minutes), seconds, per_seed)
            else:
                remaining = ""

            print('seed {} / {}: {}\t{}'.format(seed, n_seeds, probas, remaining))
    return probas


def compare_rdms(brain_rdms, model_rdvs, distance='spearmanr', n_perms=1000,
                 n_jobs=1, verbose=0):
    """
        Compute distance between RDM of each seed.

        Arguments
        -----------
        brain_rdms (2d np array):
            Reference RDM.

        model_rdvs (list):
            Candidate RDMs (as vectors).

        distance (str):
            Any distance defined in scipy.spatial.distance.

        n_jobs (int, default: 1):
            Number of jobs that can be run in parallel.

        verbose (int, default 0):
            Verbosity

        Return
        --------
        scores:
            2D array of score obtained for each seed and each model
    """

    n_seeds = len(brain_rdms)

    # Generate random permutations of the elements
    perms = nperms_indexes(root_tri_num(model_rdvs[0].shape[0]) + 1, n_perms)

    # Pre-processing
    if distance == "spearmanr":
        # Get rank (for repated value, their rank is averaged
        brain_rdms = rankdata(brain_rdms, axis=1)
        model_rdvs = rankdata(model_rdvs, axis=1)
        # Then only need to compute the pearson correlation
        distance = "pearsonr"
    elif distance == "ttest":

        # Test between values in 2 different area of the RDM
        for im, model in enumerate(model_rdvs):
            uvalues = np.unique(model)
            if len(uvalues) != 2:
                raise ValueError("Model must contain 2 different values")
            bin_model = np.zeros(model.shape, dtype=np.uint8)
            bin_model[model == uvalues[1]] = 1
            model_rdvs[im] = bin_model

    start_t = time.time()
    if distance == "ttest":
        # Compute the probabilities for each voxel in parallel
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_group_iter_numerical_comparison)(
                brain_rdms[s], model_rdvs, perms,
                verbose=max(0, verbose-1), seed=s, n_seeds=n_seeds,
                start_time=start_t)
            for s in range(n_seeds)
        )
    else:
        # Compute the probabilities for each voxel in parallel
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_group_iter_compare_rdms)(
                brain_rdms[s], model_rdvs, distance, perms,
                verbose=max(0, verbose-1), seed=s, n_seeds=n_seeds,
                start_time=start_t)
            for s in range(n_seeds)
        )

    if verbose:
        dt = time.time() - start_t
        print("Elapsed time to compare RDMs: {:.01f}s".format(dt))

    return np.array(scores)

