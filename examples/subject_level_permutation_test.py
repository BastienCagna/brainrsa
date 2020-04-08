"""
    Voice Localizer RSA analysis on a subject
    =========================================

    Load singletrial beta maps of a subject. First half of them correspond to
    voice stimuli and second half to non-voice stimuli.
    Then compute brain RDMs in the ROI and finally compare (permutation
    test) those RDMs with the Voice/Non-voice model RDM.

"""

import os.path as op
import nibabel as nb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nilearn.image import resample_to_img, concat_imgs, new_img_like, math_img
from nilearn import plotting
from sklearn import manifold

from brainrsa import SearchLightRSA
from brainrsa.rdm import check_rdm
from brainrsa.stats import mantel_test
from brainrsa.plotting import plot_rdm, plot_dist, plot_position
from brainrsa.utils.misc import check_mask
from brainrsa.utils import datasets


# ******************************************************************************
# ***** Input parameters *******************************************************
# ******************************************************************************
dataset = datasets.load_ffa_dataset()

# Subject to use
sub = "sub-01"

# Where beta are avaialable
data_mask = dataset["subjects_data"][sub]["datamask"]

# ROI mask
roi = check_mask(dataset["subjects_data"][sub]["roi"], threshold=.5)
gm = check_mask(dataset["subjects_data"][sub]["graymatter"], threshold=.5)
process_mask = math_img('(gm*roi) > 0.5', roi=roi, gm=resample_to_img(gm, roi))

# Beta maps - use only one run
beta_imgs = concat_imgs(dataset["subjects_data"][sub]["run_1"])
n_betas = len(dataset["subjects_data"][sub]["run_1"])

# Beta labels
print(dataset["beta_labels"])
labels_df =pd.read_csv(dataset["beta_labels"])


# ******************************************************************************
# ***** Compute brain RDMs in the ROI ******************************************
# ******************************************************************************
# Load the mask and check that it is binary
data_mask = check_mask(data_mask, threshold=0.5)

# Resample procmask to match datamask resolution
process_mask = check_mask(
    resample_to_img(process_mask, data_mask), 
    threshold=0.5
)

# Initialise the searchlight and specify where to compute the values and the 
# radius of the spege (in milimeters)
rsa = SearchLightRSA(
    data_mask,
    process_mask_img=process_mask,
    distance='euclidean',
    radius=6,
    n_jobs=-1,
    verbose=2
)

# Compute beta distance matrix (RDMs) for each voxel of the process_mask (if 
# data are avaialbe around it)
# The output is a list of RDMs (as vector)
print("Start to fit brain RDMs")
rdms = rsa.fit(beta_imgs)

# Compute the average RDM
avg_rdm = np.mean(rdms, axis=0)

# ******************************************************************************
# ***** Compare with Voice/Non-Voice model *************************************
# ******************************************************************************
# Create the model RDM
mid = int(n_betas/2)
model = np.zeros((n_betas, n_betas))
model[:mid, mid:] = 1
model[mid:, :mid] = 1

# Then, do the comparison with each brain RDMs (permuatation test)
# Do it on a short number of perm for test purpose
print('start to compare brain RDMs to model RDM')
scores_img = rsa.compare_to(model, distance="spearmanr", n_perms=200)

# Compute just the distance with the model
pval, true_score, random_score = mantel_test(avg_rdm, model, 
                                             "spearmanr", 1000)


# MDS
seed = np.random.RandomState(seed=3)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, 
                   random_state=seed, dissimilarity="precomputed", n_jobs=1)
pos_mds = mds.fit(check_rdm(avg_rdm)).embedding_
pos_smacof, _ = manifold.smacof(check_rdm(avg_rdm), n_components=2, 
                                random_state=seed)

# ******************************************************************************
# ***** Figure *****************************************************************
# ******************************************************************************
fig, axes = plt.subplots(2, 3, figsize=(12, 10))
# The average RDM
plot_rdm(avg_rdm, "Average RDM", triangle="lower", ax=axes[0, 0],
         cblabel="Euclidean distance")

# Plot 
plot_position(pos_mds, ax=axes[0, 1],
              title="Brain representation of stimuli in 2D space",
              names=np.array(labels_df["beta_index"], dtype=str), 
              labels=labels_df["label"])

plot_position(pos_smacof, ax=axes[0, 2],
              title="Brain representation of stimuli in 2D space (SMACOF)",
              names=np.array(labels_df["beta_index"], dtype=str), 
              labels=labels_df["label"])

# The model
plot_rdm(model, "Voice/Non-voice model", triangle="lower", ax=axes[1, 0],
         cblabel="Dissimilarity", discret=True)

# Mantel test
plot_dist(random_score, true_score, pval, ax=axes[1, 1],
          title="Correlation between average and model")

# And the score map
plotting.plot_stat_map(
    scores_img, display_mode='x', cut_coords=1, colorbar=True, axes=axes[1, 2],
    threshold=.001, title="Comparison to V/NV model")

plt.tight_layout()
plt.show()

