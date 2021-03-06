"""
    Module for Representational Similarity Analysis (Kriegeskorte 2008)
    Use the searchlight framework of nilearn
    
"""
# Authors : Bastien Cagna (bastien.cagna@univ-amu.fr)

import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from nilearn import image, masking
from nilearn._utils import check_niimg_4d, check_niimg_3d
from nilearn.image.resampling import coord_transform
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
import time

from .utils.misc import tri_num, GroupIterator
from .metrics import cross_vect_score
from .rdm import check_rdm, estimate_rdms
from .stats import compare_rdms


class SearchLightRSA(BaseEstimator):
    """ Implement RDM computation in a searchlight loop
    
     Parameters
    -----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use compute distance across samples

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means 'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False

    Representational Dissimilarity Matrices (RDMs)
    -----------------------------------------------
    RDMs are store as vector (1D array).

    """
    def __init__(self, mask_img, process_mask_img=None, radius=2.,
                 distance='euclidean', n_jobs=1, verbose=0):
        self.mask_img = mask_img
        self.process_mask_img = process_mask_img if process_mask_img else mask_img
        self.radius = radius
        self.distance = distance
        self.spheres_values = None
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.rdms = None
        self.vx_mask_coords = None
        self.ref_img = None

    def set_rdms(self, rdms):
        if self.rdms is not None:
            warnings.warn("Overwritting RDMs")
        self.rdms = check_rdm(rdms, force="vector")

    def get_rdms(self):
        return rdms

    def compute_spheres_values(self, imgs):
        start_t = time.time()

        # Force to get a list of imgs even if only one is given
        imgs_to_check = [imgs] if not isinstance(imgs, list) else imgs

        # Load Nifti images
        ref_shape = imgs_to_check[0].dataobj.shape
        ref_affine = imgs_to_check[0].affine
        imgs = []
        for img in imgs_to_check:
            # check if image is 4D
            imgs.append(check_niimg_4d(img))

            # Check that all images have same number of volumes
            if ref_shape != img.dataobj.shape:
                raise ValueError("All fMRI image must have same shape")
            if np.array_equal(ref_affine, img.affine):
                warnings.warn("fMRI images do not have same affine")
        self.ref_img = imgs[0]

        # Compute world coordinates of the seeds
        process_mask_img = check_niimg_3d(self.process_mask_img)
        process_mask_img = image.resample_to_img(
            process_mask_img, imgs[0], interpolation='nearest'
        )
        self.process_mask_img = process_mask_img

        process_mask, process_mask_affine = masking._load_mask_img(
            process_mask_img
        )
        process_mask_coords = np.where(process_mask != 0)
        self.vx_mask_coords = process_mask_coords
        process_mask_coords = coord_transform(
            process_mask_coords[0], process_mask_coords[1],
            process_mask_coords[2], process_mask_affine)
        process_mask_coords = np.asarray(process_mask_coords).T

        if self.verbose:
            print("{} seeds found in the mask".format(len(process_mask_coords)))

        # Compute spheres
        _, A = _apply_mask_and_get_affinity(
            process_mask_coords, imgs[0], self.radius, True,
            mask_img=self.mask_img
        )

        # Number of runs: 1 4D fMRI image / run
        n_runs = len(imgs)

        # Number of spheres (or seed voxels)
        n_spheres = A.shape[0]

        # Number of volumes in each 4D fMRI image
        n_conditions = imgs[0].dataobj.shape[3]

        mask_img = check_niimg_3d(self.mask_img)
        mask_img = image.resample_img(
            mask_img, target_affine=imgs[0].affine,
            target_shape=imgs[0].shape[:3], interpolation='nearest'
        )

        masked_imgs_data = []
        for i_run, img in enumerate(imgs):
            masked_imgs_data.append(masking._apply_mask_fmri(img, mask_img))

        # Extract data of each sphere
        # X will be #spheres x #run x #conditions x #values
        X = []
        for i_sph in range(n_spheres):
            # Indexes of all voxels included in the current sphere
            sph_indexes = A.rows[i_sph]

            if len(sph_indexes) == 0:
                # Append when no data are available around the process voxel
                X.append(np.full((n_runs, n_conditions, 1), 0))
                print("Empty sphere")
            else:
                # Number of voxel in the current sphere
                n_values = len(sph_indexes)

                sub_X = np.empty((n_runs, n_conditions, n_values), dtype=object)
                for i_run, img in enumerate(imgs):
                    for i_cond in range(n_conditions):
                        sub_X[i_run, i_cond] = masked_imgs_data[i_run][i_cond][
                            sph_indexes]
                X.append(sub_X)

        if self.verbose:
            dt = time.time() - start_t
            print("Elapsed time to extract spheres values: {:.01f}s".format(dt))

        self.spheres_values = X

    def fit(self, imgs=None, spheres_values=None):
        """

        Parameters
        -----------
        imgs: single 4D Nifti image or list of Nifti 4D images
            Give 1  4D Nifti image per run
        
        Return
        -------
        mat: 2d-array
            RDM values
        """

        if imgs:
            if self.spheres_values:
                warnings.warn("Recomputing sphere values")
            self.compute_spheres_values(imgs)
        elif self.spheres_values is None:
            if spheres_values is None:
                raise ValueError("If no previously sphere_values are given, "
                                 "fMRI imgs are needed")
            self.spheres_values = spheres_values

        # Estimate RDMs
        mat = estimate_rdms(self.spheres_values, distance=self.distance,
                            n_jobs=self.n_jobs, verbose=max(0, self.verbose-1))
        
        self.rdms = mat
        return mat

    def distance_to(self, candidate_rdm, distance, outputAsNifti=False):
        n_seeds = len(self.rdms)
        candidate_rdm = check_rdm(candidate_rdm, force="vector")

        # Compute anly the true distance
        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(cross_vect_score)(
                self.rdms[s], candidate_rdm, scoring=distance)
            for s in range(n_seeds)
        )

        if outputAsNifti:
            # Load process mask to map the scores
            proc_mask = image.load_img(self.process_mask_img)
            mask_idx = np.argwhere(proc_mask.dataobj)
            
            # Map scores of each model to a new Nifti image
            scores_map = np.zeros(proc_mask.dataobj.shape, dtype=float)
            # TODO: do this without for loop?
            for s in range(mask_idx.shape[0]):
                a, b, c = mask_idx[s]
                scores_map[a, b, c] = scores[s]
            return image.new_img_like(proc_mask, scores_map)
        return scores

    def compare_to(self, models, normalize=False, distance='spearman',
                   n_perms=1000, outputImage=True):
        """ Compare brain activation RDMs to models
        
        Parameters
        ----------
        models: 2darrays or list 2darrays
            One or several model RDMs.
        
        normalize: boolean (default: False)
            Specify if RDM values must be noramlize between 0 and 1 before been
            used.
            
        distance: str (default: None)
            Name of the function used compute distance between RDMs
            By default, it take the one defined at the init (the same used for
            RDMs computation)
            
        n_perms: int (default: 1000)
            Number of permutation to compute the empirical random distribution
            
        Returns
        --------
        scores_imgs: list of 3D Nifti images
            Return one Nifti image by model. Each image contain the distance
            between the voxel the model.
         
        """
        start_t = time.time()
        
        # Check that RDM are given
        if self.rdms is None:
            raise ValueError("You need to fit brain RDMs first.")
            
        # If only one model is given, convert it to a list
        models = [models] if not isinstance(models, list) else models
        models = check_rdm(models, multiple=True, force="vector")

        # Compute scores of each model at each voxel
        scores = compare_rdms(self.rdms, models, distance, n_perms,
                              verbose=self.verbose, n_jobs=self.n_jobs)

        if outputImage:
            # Load process mask to map the scores
            proc_mask = image.load_img(self.process_mask_img)
            mask_idx = np.argwhere(proc_mask.dataobj)
            # if mask_idx.shape[0] != scores.shape[0]:
            #     warnings.warn("mask shape: {}   scores shape: {}".format(
            #         mask_idx.shape[0], scores.shape[0]))
            #     raise ValueError("Inconsistent mask and scores dimensions")
            
            # Map scores of each model to a new Nifti image
            scores_imgs = []
            for im in range(len(models)):
                scores_map = np.zeros(proc_mask.dataobj.shape, dtype=float)
                # TODO: do this without for loop?
                for s in range(mask_idx.shape[0]):
                    a, b, c = mask_idx[s]
                    scores_map[a, b, c] = scores[s, im]
                scores_imgs.append(image.new_img_like(proc_mask, scores_map))

            if self.verbose:
                dt = time.time() - start_t
                print("Elapsed time to compare to {} model(s): {:.01f}s".format(
                    len(models), dt))
            
            # Return Nifti images
            return scores_imgs if len(scores_imgs) > 1 else scores_imgs[0]
        return scores if len(scores) > 1 else scores[0]

    def index_image(self):
        ind_map = - np.ones(check_niimg_4d(self.ref_img).shape[:3])
        for i, (x, y, z) in enumerate(np.array(self.vx_mask_coords).T):
            ind_map[x, y, z] = i
        return image.new_img_like(self.ref_img, ind_map)

