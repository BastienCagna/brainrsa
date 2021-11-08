from brainrsa.rdm import check_rdm
from brainrsa.plotting import plot_rdm
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt


def clusterize_rdm(rdm, K):
    rdm = check_rdm(rdm, force="matrix")
    n_obj = rdm.shape[0]

    vect_rdm = check_rdm(rdm, force="vector")
    print(np.sum(vect_rdm == 0))
    X = np.atleast_2d(vect_rdm[vect_rdm != 0]).T

    print("Clusterization K={}".format(K))
    model = KMeans(n_clusters=K, random_state=1)
    #model = BayesianGaussianMixture(n_components=K, random_state=1, covariance_type="tied")
    pred_rdm = model.fit_predict(X)

    fig = plt.figure(figsize=(12, 7))
    cmap = plt.cm.get_cmap('Set1', K)

    ax = plt.subplot(2, 2, 1, frameon=False)
    plot_rdm(rdm, fig=fig, ax=ax, title="Original RDM")

    plt.subplot(2, 2, 2)
    bins = np.linspace(np.min(X), np.max(X), 10)
    for k in range(K):
        plt.hist(X[pred_rdm == k, 0], bins=bins, color=cmap(k/K), alpha=0.6,
                 label="cluster {}".format(k))
    plt.title("Histogram")
    plt.legend(loc="upper right")
    plt.grid()

    ax = plt.subplot(2, 2, 3)
    plot_rdm(pred_rdm, discret=True,
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
    dendrogram(linkage(dist_mat), labels=list(
        "cluster {}".format(k) for k in range(K)))
    plt.tight_layout()
    return


if __name__:
    clusterize_rdm()
