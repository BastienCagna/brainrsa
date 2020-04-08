#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   Metrics used to compute RDMs

   @author: Bastien Cagna
"""

import numpy as np
from scipy.stats.mstats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean, mahalanobis


def cross_vect_score(vect_a, vect_b, scoring='euclidean', inv_noise_cov=None):
    """ Use the scoring function to compute a value between two vectors

    Parameters
    ----------
    vect_a, vect_b: vector
        Data vectors.

    scoring:
        Scoring function in euclidean / mahalanobis / crossnobis / spearmanr /
        pearsornr. If "spearmanr_dist", return 1 - spearmanr correlation.

    inv_noise_cov: 2D array
        Inverse of the noise covariance matrix needed for mahalanobis and
        crossnobis scorings.

    Returns
    -------
    score: float
        Score value.
        
    """
    if scoring == 'euclidean':
        score = euclidean(vect_a, vect_b)
    elif scoring == "mahalanobis":
        score = mahalanobis(vect_a, vect_b, inv_noise_cov)
    elif scoring == "crossnobis":
        raise NotImplemented("Cross validated Mahalanobis distance is not " + \
                             "yet available")
    elif scoring in ["spearmanr", "spearmanr_dist"]:
        # Warning: ranking takes time, it's faster to input ranked vectors and
        # use pearsonr distance when doing multiple test on same vectors
        score, _ = spearmanr(vect_a, vect_b)
    elif scoring == "pearsonr":
        score, _ = pearsonr(vect_a, vect_b)
    else:
        raise ValueError("Unknown scoring function")

    if scoring[-5:] == "_dist":
        return 1 - score
    return score

