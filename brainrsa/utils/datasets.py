"""
    Support function for loading test datasets
"""
import os
import os.path as op
from os import makedirs
import pandas as pd
import nibabel as nb


def load_data(name, dname, path_to=None):
    if path_to is None:
        path_to = op.join(op.expanduser("~"), ".data")
    
    data_dir = op.join(path_to, "brainrsa")
    
    makedirs(data_dir, exist_ok=True)

    data_path = op.join(data_dir, dname)
    if op.exists(data_path):
        print("{} Already exists, skipping download".format(data_path))
        return data_path

    data_zip = op.join(data_dir, "{}.zip".format(name))

    print("Download {}".format(data_zip))
    oc_path = "https://amubox.univ-amu.fr/s/{}/download".format(name)
    os.system("wget -O {} --no-check-certificate  --content-disposition\
        {}".format(data_zip, oc_path))

    if not op.exists(data_zip):
        raise IOError("Download failed")

    print("Unzip {} in {}".format(data_zip, data_path))
    os.system("unzip -o {} -d {}".format(data_zip, data_path))
    os.remove(data_zip)
    
    if not op.exists(data_path):
        print("Unzip failed")
    return data_path


def load_ffa_dataset(path_to=None):
    """ Load test data, template and needed scripts """
    data_path = load_data("jdtMw9fs8trB83a", "ffa_dataset", path_to)

    beta_labels = pd.read_csv(op.join(data_path, "beta_labels.csv"))
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

