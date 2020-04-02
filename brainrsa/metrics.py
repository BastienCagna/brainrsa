import numpy as np
from scipy.stats.mstats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean

def cross_vect_score(rdm_a, rdm_b, scoring='euclidean'):
    """
        TODO: add doc !
    """
    if scoring == 'euclidean':
        score = euclidean(rdm_a, rdm_b)#np.mean(rdm_b - rdm_a)
#    elif scoring in ['correlation', "correlation_dist"]:
#        a = np.sqrt(np.sum(np.power(rdm_a, 2)))
#        b = np.sqrt(np.sum(np.power(rdm_b, 2)))
#        score = np.dot(rdm_a, rdm_b) / (a * b)
    elif scoring in ["spearmanr", "spearmanr_dist"]:
        # Warning: ranking takes time, it's faster to input ranked vectors and
        # use pearsonr distance when doing multiple test on same vectors
        score, _ = spearmanr(rdm_a, rdm_b)
    elif scoring == "pearsonr":
        score, _ = pearsonr(rdm_a, rdm_b)
    else:
        raise ValueError("Unknown scoring function")

    if scoring[-5:] == "_dist":
        return 1 - score
    return score
