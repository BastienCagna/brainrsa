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
import numpy as np
import matplotlib.pyplot as plt

from nilearn.image import resample_to_img, concat_imgs, new_img_like
from nilearn import plotting

from brainrsa import SearchLightRSA
from brainrsa.stats import mantel_test
from brainrsa.plotting import plot_rdm, plot_dist
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
process_mask = dataset["subjects_data"][sub]["roi"]

# Beta maps
n_betas = len(dataset["subjects_data"][sub]["run_1"])
imgs = []
for i in range(n_betas):
    b1 = dataset["subjects_data"][sub]["run_1"][i]
    b2 = dataset["subjects_data"][sub]["run_2"][i]
    dt = (np.array(nb.load(b1).dataobj) + np.array(nb.load(b2).dataobj)) / 2
    imgs.append(new_img_like(b1, dt))

print("Using data of {} ({} betas)".format(sub, n_betas))

# Then, load all the images and put them in a new one (4D)
beta_imgs = concat_imgs(imgs)


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
print('start to compare brain RDMs to model RDM')
scores_img = rsa.compare_to(model, distance="spearmanr", n_perms=100)

# Compute just the distance with the model
pval, true_score, random_score = mantel_test(avg_rdm, model, 
                                             "spearmanr", 10000)

# ******************************************************************************
# ***** Figure *****************************************************************
# ******************************************************************************
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# The average RDM
plot_rdm(avg_rdm, "Average RDM", sigtri="lower", ax=axes[0, 0])

# The model
plot_rdm(model, "Voice/Non-voice model", sigtri="lower", ax=axes[0, 1])

# Mantel test
plot_dist(random_score, true_score, pval, ax=axes[1, 0],
          title="Correlation between average and model")

# And the score map
plotting.plot_stat_map(
    scores_img, display_mode='x', cut_coords=1, colorbar=True, axes=axes[1, 1], 
    title="Comparison to V/NV model")

plot_rdm(avg_rdm, "Average RDM")
plotting.show()

