import nibabel as nib
import numpy as np
from searchlight.searchlight import RSASearchLight
import pyrsa

from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

import time
from tqdm import tqdm
import os.path as op
from os import mkdir
from shutil import rmtree
from nilearn.image import resample_to_img, new_img_like
import nilearn.plotting as nplt


def main():
    # Parameters
    # ==========================================================================
    # Global parameters
    n_conditions = 40
    njobs = 20

    # RDM estimation parameters
    radius = 2
    roiname = "brain"
    searchlight_thd = .7

    # Inference parameters
    inf_distance = 'spearmanr'
    n_perms = 100
    inf_thd = 0.01

    # Data paths
    s = "sub-04"
    sub_dir = op.join("/hpc/banco/cagna.b/projects/ageing/data/preprocessing", s)
    glm_d = op.join(sub_dir, "glm", "ua" + s +"_task-voicelocalizer_model-singletrial")
    mask_f = op.join(sub_dir, 'anat', s + '_brainmask.nii')
    t1_f = op.join(sub_dir, 'anat', 'sanlm_' + s + '_T1w.nii')
    out_d = op.join(sub_dir, 'rsa', 'ua{}_task-voicelocalizer_roi-{}_radius-{}vx'.format(s, roiname, radius))

    # Initialisation
    # ==========================================================================
    # (Re-)initialize output directory
    if op.isdir(out_d):
        rmtree(out_d)
    mkdir(out_d)

    # Load betas
    # ==========================================================================
    print("Loading beta maps")
    # Load a beta map to get reference affine and shape
    beta_ref = nib.load(op.join(glm_d, "beta_0001.nii"))
    x, y, z = beta_ref.shape

    # Load and put all betas in a single numpy array
    betas = np.empty((x, y, z, n_conditions))
    for i in tqdm(range(n_conditions), ncols=60):
        beta = nib.load(op.join(glm_d, "beta_{:04d}.nii".format(i+1)))
        betas[:, :, :, i] = np.array(beta.dataobj)

    # Create mask: intersection between brain mask and GLM mask
    # ==========================================================================
    # Resample brain mask to match the beta maps dimensions
    brainmask = resample_to_img(mask_f, beta_ref)

    # GLM mask have already the same dimension of beta maps
    glm_mask = nib.load(op.join(glm_d, "mask.nii")).dataobj

    # Compute intersection
    mask = np.array(brainmask.dataobj)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    mask *= glm_mask

    # Save it
    nib.save(new_img_like(brainmask, mask), op.join(out_d, "mask.nii"))

    # Compute RDMs
    # ==========================================================================
    print("Init searchlight")
    # initialise the searchlight.
    SL = RSASearchLight(
        mask,
        radius=radius,
        threshold=searchlight_thd,
        njobs=njobs,
        verbose=True
    )

    print("Fit brain RDMs")
    # Compute Brain RDMs
    SL.fit_rsa(betas, wantreshape=False)
    np.save(op.join(out_d, "RDMs.npy"), SL.RDM)

    # Input RDMs in pyrsa framework
    rdms = pyrsa.rdm.RDMs(np.array(SL.RDM))
    print(rdms.get_matrices().shape)

    # Load models
    # ==========================================================================
    # Voice vs. Non-voice model
    mid = int(n_conditions/2)
    vnv_model = np.zeros((n_conditions, n_conditions))
    vnv_model[:mid, mid:] = 1
    vnv_model[mid:, :mid] = 1
    vnv_model = squareform(vnv_model)

    # Onset model
    # onsets_rdm = np.load(op.join( 'models', 'rdm_onsets_int.npy'))

    # Do inference
    # ==========================================================================
    print('start to compare brain RDMs to model RDMs')
    scores = compare_rdms(rdms.get_vectors(), [vnv_model], inf_distance,
                          n_perms, verbose=True, n_jobs=njobs)


if __name__ == "__main__":
    # Running this script will run the main() function
    main()

