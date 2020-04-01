"""
    Voice Localizer RSA analysis on a subject
    =========================================

    Load singletrial beta maps of a subject. First half of them correspond to
    voice stimuli and second half to non-voice stimuli.
    Then compute brain RDMs in the grey matter and finally compare (permutation
    test) those RDMs with the Voice/Non-voice model RDM.

"""

import os.path as op
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt

from nilearn.image import resample_to_img, concat_imgs
from nilearn import plotting

from brainrsa import SearchLightRSA
from brainrsa.plotting import plot_rdm
from brainrsa.utils import check_mask


# ******************************************************************************
# ***** Input parameters *******************************************************
# ******************************************************************************
# Number of betas (to load them and also construct the model RDM)
n_betas = 40

# Will compute brain RDMs only in grey matter
process_mask = "/hpc/banco/cagna.b/projects/ageing/data/preprocessing/" + \
               "sub-04/anat/c1sanlm_sub-04_T1w.nii"
# Directory of the singletrial GLM
beta_dir = "/hpc/banco/cagna.b/projects/ageing/data/analyses/sub-04/glm/" + \
           "uasub-04_task-voicelocalizer_model-singletrial"
# Where the beta value sare actually available
data_mask = op.join(beta_dir, "mask.nii")

# ******************************************************************************
# ***** Compute brain RDMs in the ROI ******************************************
# ******************************************************************************
# List beta files of the subject
beta_imgs = []
for b in range(n_betas):
    beta_imgs.append(op.join(beta_dir, "beta_{:04d}.nii".format(b+1)))

# Then, load all the images and put them in a new one (4D)
beta_imgs = concat_imgs(beta_imgs)

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
    n_jobs=20,
    verbose=2
)

# Compute beta distance matrix (RDMs) for each voxel of the process_mask using
# voxels included in data_mask
print("Start to fit brain RDMs")
rdms = rsa.fit(beta_imgs)

# Save the image that give RDM index at each voxel
# nb.save(rsa.index_image(), indx_map)

# Save RDMs
# np.save(dist_f, rdms)

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
#corr_img = rsa.distance_to(squareform(models[im]), distance, True)
#save_map(corr_img, brain, dist2model_tmpl.replace("[m]", model),
#         "Spearman ranked distance to " + model + " model")

# ******************************************************************************
# ***** Figure *****************************************************************
# ******************************************************************************
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Show the mask
plotting.plot_roi(process_mask, data_mask, display_mode='x', cut_coords=1, 
                  axes=axes[0, 0], title="Process mask over data mask")

# The average RDM
plot_rdm(avg_rdm, "Average RDM", sigtri="lower", ax=axes[0, 1])

# The model
plot_rdm(model, "Voice/Non-voice model", sigtri="lower", ax=axes[0, 1])

# And the score map
plotting.plot_stat_map(
    scores_img, display_mode='x', cut_coords=1, colorbar=True, axes=axes[1, 1], 
    title="Comparison to V/NV model")

plotting.show()
