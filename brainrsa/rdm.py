"""


"""

import os.path as op
import nibabel as nb
import argparse

import numpy as np
import pandas as pd
import math as m
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from nilearn.image import resample_to_img, math_img
from nilearn.masking import apply_mask
from scipy.stats.mstats import spearmanr
from scipy.spatial.distance import squareform, euclidean
from joblib import Parallel, delayed

from warnings import warn
import time
import matplotlib.pyplot as plt

from .utils.misc import root_tri_num, tri_num, GroupIterator
from .metrics import cross_vect_score


def check_rdm(rdm, multiple=None, **kwargs):
    """ Interface to _check_rdm to handle also list of RDMs.

    Arguments
    =========
    rdm: 
        see _check_rdm()
    
    multiple: bool or None
        If True, read rdm as a list of rdms (either as vector or matrix).
        If False, rdm is a single RDM. If None, if rdm is a 3D array or that 
        two first dimension are different, assume that each row is an RDM.
    
    **kwargs:
        see _check_rdm()

    Return
    ======
    One or a list of RDM.
    """
    rdm = np.array(rdm)
    if multiple or (len(rdm.shape) > 1 and rdm.shape[0] != rdm.shape[1]):
        rdms = []
        for sub_rdm in rdm:
            rdms.append(_check_rdm(sub_rdm, **kwargs))
        return np.array(rdms)
    else:
        return _check_rdm(rdm, **kwargs)    
    

def _check_rdm(rdm, force="matrix", sigtri="both", include_diag=False, 
              fill=None, norm=None, vmin=None, vmax=None):
    """
        Usefull function force the represention of the RDM to be either as 
        matrix (default) or as vector.

        Can also be used to mask upper, lower or just diagonal values with as
        given value (np.nan by default).

        Arguments
        =========
        rdm: 1d or 2d array-like
            RDM as vector or matrix.

        force: "matrix" or "vector"
        
        sigtri: "upper", "lower", "both" or None
            If sigtri is "both", the upper and lower triangle must be redundant. 
            Otherwise the upper or lower triangle is assumed to contain the
            values.

        include_diag: bool
            If False, the diagonal is included in the mask. Otherwise, digonal 
            value are not modified (basically kept to 0).

        norm: "upper", "lower", "both" or None
            If None, no normalization is performed.
    
        vmin, vmax: int or float
            Must be defined if norm is not None.

        Return
        ======
        rdm

    """
    rdm = np.array(rdm)
    
    is_matrix = len(rdm.shape) == 2

    if is_matrix and rdm.shape[0] != rdm.shape[1]:
        raise ValueError("Non square matrix given.")

    # If the RDM is a matrix and that she supposed to be a redeondent distance 
    # matrix, verify that upper and lower triangle are equal.
    if sigtri == "both" and is_matrix:  
        ucoords = np.triu_indices(rdm.shape[0], k=1 if not include_diag else 0)
        upper_vals = rdm[ucoords]
        lower_vals = rdm.T[ucoords]
        if not np.array_equal(upper_vals, lower_vals):
            warn(
                "RDM matrix is supposed to be a redundant distance matrix " \
                '(sigtri="both") but upper and lower triangles are not equal.'
            )
            rdm = check_rdm(rdm, "vector", None, include_diag)
            is_matrix = False

    if force == "matrix":
        if rdm.dtype == float or ((not include_diag or sigtri != "both") and type(fill) == float):
            dtype = float
        else:
            dtype = int
        
        # If the RDM is in vector form
        if not is_matrix:
            # Convert to matrix
            rdm = squareform(rdm)

        # If vmin and vmax are defined, fill with normalized rdm
        if norm is not None:
            rdm = normalized_rdm(rdm, norm, vmin, vmax)

        if dtype != rdm.dtype:
            rdm = rdm.astype(dtype)
        
        if fill is not None:
            if sigtri == "lower":
                ix = np.triu_indices(rdm.shape[0], k=1 if include_diag else 0)
                rdm[ix] = fill
            elif sigtri == "upper":
                ix = np.tril_indices(rdm.shape[0], k=-1 if include_diag else 0)
                rdm[ix] = fill
            else:
                rdm[np.eye(rdm.shape[0])==1] = fill
            
    elif force=="vector":
        # If the RDM is in matrix form
        if is_matrix:
            # Take always the upper triangle to list the values such the order
            # is the same than when going from vector to matrix
            if sigtri == "lower":
                rdm = rdm.T
            elif sigtri is None:
                warn("As sigtri=None, take values from the upper triangle to " \
                     "convert to vector form.")
            dtype = rdm.dtype
            ix = np.triu_indices(rdm.shape[0], k=1 if not include_diag else 0)
            
            rdm = rdm[ix].astype(dtype)
    else:
        raise ValueError('Unknown force="' + str(force) + '"')
    return rdm
       

def estimate_rdms(X, distance='euclidean', n_jobs=1, verbose=0):
    """ Compute a RDM for if set of value of X
    
    Return
    ======
    rdms:
        Nbr seed x Nbr elements
    """
    n_seeds = len(X)
    
    start_t = time.time()

    group_iter = GroupIterator(n_seeds, n_jobs)
    rdms = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_group_iter_rdm)(
            X, seed_list, distance, thread_id, n_seeds, max(0, verbose-1))
        for thread_id, seed_list in enumerate(group_iter)
    )
    rdms = np.concatenate(rdms)

    if verbose:
        dt = time.time() - start_t
        print("Elapsed time for RDMs computation: {:.01f}s".format(dt))
    return np.array(rdms)


def _group_iter_rdm(X, seed_list, distance, thread_id, total, verbose=0):
    rdms = []
    for s in seed_list:
        rdms.append(_rdm_job(X[s], distance, s, total, verbose))
    return rdms


def _rdm_job(Xseed, distance, seed_id, total, verbose=0):
    """ Compute RDM
    
    The dissimilarity can be computed with various metrics as euclidean
    distance or spearman (pearson ranked) correlation.
    """
    ncond = Xseed.shape[1]
    nvoxels = Xseed.shape[2]
    
    # TODO: add cross validation here ?
    # Average all runs
    Xmean = np.mean(Xseed, axis=0)
    
    if verbose:
        print("sphere {}/{} has {} values".format(seed_id, total, n_voxels))

    # Rank values for each beta to speed up the computation
    if distance == "spearmanr":
        Xmean = np.argsort(Xmean, axis=1)
        distance = "pearsonr"

    n_elem = tri_num(ncond-1)
    rdm_v = np.empty((n_elem,), dtype=float)
    i_elem = 0
    for i in range(ncond-1):
        for j in range(i+1, ncond):
            rdm_v[i_elem] = cross_vect_score(Xmean[i], Xmean[j], distance)
            i_elem += 1
    return rdm_v



def normalized_rdm(rdm, norm, vmin, vmax):
    """

    """
    if norm not in ["both", "upper", "lower"]:
        raise ValueError('Unknown behaviour for norm="' + str(norm) + '"')

    rdm = check_rdm(rdm)
    
    if vmin is None or np.isnan(vmin) or vmax is None or np.isnan(vmax):
        raise ValueError('vmin and/or vmax is/are not a number.')

    norm_rdm = (rdm - vmin) / (vmax - vmin)

    if norm_rdm.max() > 1 or norm_rdm.min() < 0:
        warn("Some values of the RDM are beyond the given range " + \
             "[{}, {}]".format(vmin, vmax))
        norm_rdm[norm_rdm>1] = 1
        norm_rdm[norm_rdm<0] = 0

    if norm == "both":
        return norm_rdm

    rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min())

    n = rdm.shape[0]
    i = np.triu_indices(n, k=1) if norm == "upper" else np.tril_indices(n, k=-1)
    rdm[i] = norm_rdm[i]
    return rdm
        

#### CONSTRUCT RDM



def age_model_rdm(participant_tsv):
    """
    Parameters
    ============
    subs: list of str
        List of subjects.
    participant_tsv: str
        Path to participant.tsv to read ages.

    Returns
    ========
    rdm: 2d-array
        Ageing model RDM.
    subject_order: list

    """
    # Read age of each participant
    participants = pd.read_csv(participant_tsv, sep=",")
    subs = np.array(participants["subject"])
    ages = np.array(participants["age"])

    # Order participant by ascending age
    subs_order = np.argsort(ages)
    subs = subs[subs_order]
    ages = ages[subs_order]

    print("Age group head counts:")
    age_bounds = []
    for start in range(0, 90, 10):
        count = np.sum((ages>=start) * (ages<(start+10)))
        print("\t[{}, {}]: {} subjects".format(start, start+9, count))
        if len(age_bounds) == 0:
            age_bounds.append(count)
        else:
            age_bounds.append(age_bounds[-1] + count)
    
    # Construct age RDM
    ages_2d = np.repeat(np.atleast_2d(ages), len(ages), axis=0)
    rdm = np.abs(ages_2d - ages_2d.T)
    return rdm, subs, np.array(age_bounds), ages


def roi_average_rdm(rdms_f, rdm_idx_f, roi_f):
    rdms = np.load(rdms_f)

    # Mask index map with the ROI
    idx_img = nb.load(rdm_idx_f)
    roi_img = math_img('img > 0.5', img=resample_to_img(roi_f, idx_img))
    try:
        idx_sel = apply_mask(idx_img, roi_img)
        idx_sel = np.array(idx_sel[idx_sel > -1], dtype=int)
    except ValueError:
        print("************ problem with: ******************")
        print(rdms_f)
        print(rdm_idx_f)
        print(roi_f)
        return np.zeros(rdms[0].shape)
    
    # Load RDMs and average those included in the ROI    
    return np.mean(rdms[idx_sel], axis=0)
       

def clusterize_rdm(rdm, K):
    rdm = check_rdm(rdm, force="matrix")
    n_obj = rdm.shape[0]
    
    vect_rdm = check_rdm(rdm, force="vector")
    print(np.sum(vect_rdm==0))
    X = np.atleast_2d(vect_rdm[vect_rdm!=0]).T

    print("Clusterization K={}".format(K))
    model = KMeans(n_clusters=K, random_state=1)
    #model = BayesianGaussianMixture(n_components=K, random_state=1, covariance_type="tied")
    pred_rdm = model.fit_predict(X)

    fig = plt.figure(figsize=(12, 7))
    cmap = plt.cm.get_cmap('Set1', K)

    ax = plt.subplot(2, 2, 1, frameon=False)
    plot(rdm,fig=fig, ax=ax, title="Original RDM")

    plt.subplot(2, 2, 2)
    bins = np.linspace(np.min(X), np.max(X), 10)
    for k in range(K):
        plt.hist(X[pred_rdm==k, 0], bins=bins, color=cmap(k/K), alpha=0.6,
                 label="cluster {}".format(k))
    plt.title("Histogram")
    plt.legend(loc="upper right")
    plt.grid()

    ax = plt.subplot(2, 2, 3)
    plot(pred_rdm, discret=True,
             title="Clusterized RDM (K={})".format(K), fig=fig, ax=ax)

    plt.subplot(2, 2, 4)
    plt.title('Clusters Dendrogram')
    centers = model.cluster_centers_
    dist_mat = np.zeros((K, K))
    for c1 in range(K):
        for c2 in range(c1):
            d = euclidean(centers[c1], centers[c2])
            dist_mat[c1, c2] = d
            dist_mat[c2, c1] = d
    dendrogram(linkage(dist_mat), labels=list("cluster {}".format(k) for k in range(K)))
    plt.tight_layout()
    return 


def test():
    vect = np.arange(10).astype(int)
    labels = list("obj {}".format(i) for i in range(5))
    print(vect)

    rdm = check_rdm(vect, force="matrix", sigtri="upper", fill=np.inf)
    print(rdm)

    print(check_rdm(rdm, force="vector", sigtri=None))

    print(check_rdm(vect, norm="lower", vmin=0, vmax=100))

    vect = check_rdm(rdm, force="vector", sigtri="upper")
    print(vect)
    plot(vect, norm="upper", labels=labels, vmin=0, vmax=20, include_diag=True, title="Test RDM")
    plt.show()




if __name__ == "__main__":
    test()
    
