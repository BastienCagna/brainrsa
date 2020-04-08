"""
    Support function for loading test datasets
"""
import os
import os.path as op
from os import makedirs
import numpy as np
import pandas as pd
import nibabel as nb
from glob import glob


def load_data(url, dname, ext=None, path_to=None):
    # If no othe rpath is given, download in the user's home directory
    if path_to is None:
        path_to = op.join(op.expanduser("~"), ".data")
 
    data_dir = op.join(path_to, "brainrsa")
 
    # Verify that the data directory already exist   
    makedirs(data_dir, exist_ok=True)

    # Path to the dataset's directory
    data_path = op.join(data_dir, dname)

    # If it already exist, skip the download
    if op.exists(data_path):
        print("{} Already exists, skipping download".format(data_path))
        return data_path

    # filename
    if ext:
        data_f = op.join(data_dir, "{}.{}".format(dname, ext))
    else:
        data_f = data_path
        makedirs(op.split(data_path)[0], exist_ok=True)

    # Download from the given URL
    print("Download {}".format(data_f))
    os.system("wget -O {} --no-check-certificate  --content-disposition\
        {}".format(data_f, url))

    if not op.exists(data_f):
        raise IOError("Download failed")

    if ext == "gz":
        print("Unzip {} in {}".format(data_f, data_path))
        os.system("unzip -o {} -d {}".format(data_f, data_path))
        os.remove(data_f)
    elif ext == "tgz":
        print("Uncompress {} in {}".format(data_f, data_path))
        os.system("tar -zxf {} -C {}".format(data_f, op.split(data_path)[0]))
        os.remove(data_f)

    return data_path


def load_ffa_dataset(path_to=None):
    """ Load test data, template and needed scripts """
    url = "https://amubox.univ-amu.fr/s/jdtMw9fs8trB83a/download"
    data_path = load_data(url, "ffa_dataset", "gz", path_to)

    beta_labels = op.join(data_path, "beta_labels.csv")
    subs = {}
    for f in os.listdir(data_path):
        if f[:4] == "sub-":
            sub = f
            beta_maps = []
#            for ff in os.listdir(op.join(data_path, f)):
#                if (sub + "_beta_") in ff:
#                    beta_maps.append(op.join(data_path, f, ff))

            dt = {}
            dt["run_1"] = list(op.join(data_path, f, sub + "_beta_{:04d}.nii.gz".format(i)) for i in range(1, 13))
            dt["run_2"] = list(op.join(data_path, f, sub + "_beta_{:04d}.nii.gz".format(i+12)) for i in range(1, 13))
            for k in ["brain", "datamask", "graymatter", "roi"]:
                dt[k] = op.join(data_path, sub, sub + "_" + k + ".nii.gz")
            subs[sub] = dt
    return {"beta_labels": beta_labels, "subjects_data": subs}


def load_intertva_dataset(sub=None, path_to=None):
    av_subnums = np.concatenate((range(3, 36), range(37, 43)))
    av_subs = list("sub-{:02d}".format(snum) for snum in av_subnums)
    
    if sub is None:
        sub = av_subs
    elif isinstance(sub, str):
        sub = [sub]

    data = {}
    url = "https://zenodo.org/record/2591038/files/labels_voicelocalizer_voice_vs_nonvoice.tsv?download=1"
    data["labels"] = load_data(url, op.join("intertva_dataset", "labels.tsv"), path_to)
    url = "https://zenodo.org/record/2591038/files/brain_mask.nii.gz?download=1"
    data["brainmask"]= load_data(url, op.join("intertva_dataset", "brain_mask.nii.gz"), path_to)
    
    for s in sub:
        if not s in av_subs:
            raise ValueError(s + " is not a valid subject's ID." \
                             "Available  subjects are: {}".format(av_subs))
        url =  "https://zenodo.org/record/2591038/files/InterTVA_" + s + ".tgz?download=1"
        data_path = load_data(url, op.join("intertva_dataset", s), "tgz", path_to)
        data[s] = glob(op.join(data_path, "*"))
    return data

