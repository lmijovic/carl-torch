

from ml import RatioEstimator
# modified loader for generic dataframe
from ml import Loader_edb

#
from variables import get_variable_names
from utils_edb import preprocess_inputs

# support for patch parsing in hyperpara optization
import json
import argparse
from shutil import copy2

import os
import sys
import logging
import optparse
import torch
import tarfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------
# main 
# reference sample which gets the weights assigned
infile_old="ref.csv"
# sample to which we want to weight
infile_new="to_weight.csv"

# where outputs go (make sure these dirs exist)
data_out_path="data"
store_data=True
model_out_path="model"

# try to use same seed in all steps for reproducibility
random_seed=42

#-----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Predict carl weights')
parser.add_argument('--patch', required=False, help='hyperparameters patch in .json format', default='')

# default hyperparams
# hidden layers nodes 
n_hidden=(9,9,9)
n_epochs=200

args = parser.parse_args()
if (args.patch!=""):
    nodevals=[]
    with open(args.patch) as f:
        patch_hyperparas=json.load(f) 
        print('applying hyperparameter patch: ', patch_hyperparas)
        for k,v in patch_hyperparas.items():
            if ('epochs'==k):
                n_epochs=int(v)
            elif ('nodes'==k):
                for nv in v.split(","):
                    nodevals.append(int(nv))
                n_hidden=tuple(nodevals)
            else:
                print("Warning, the patch contains unknown key",k)

if os.path.exists(data_out_path + '/X_train.npy'):
    x=data_out_path+'/X_train.npy'
    y=data_out_path+'/y_train.npy'
    x0=data_out_path+'/X0_train.npy'
    x1=data_out_path+'/X1_train.npy'
    print("Loaded existing datasets ")
    if torch.cuda.is_available():
        tar = tarfile.open("data_out.tar.gz", "w:gz")
        for name in [data_out_path +'/X0_train.npy']:
            tar.add(name)
        tar.close()
else:
    # read variables from csv and map them to columns:
    col_names = get_variable_names()
    # read-in csv
    # in case 1st row contains column pseudo-names, set skiprows=1
    data_x0=pd.read_csv(infile_old,skiprows=0, header=None,names=col_names)
    data_x1=pd.read_csv(infile_new,skiprows=0, header=None,names=col_names)

    # preprocessing of generic input to ensure they satisfy carl-torch framework req.
    data_x0,data_x1=preprocess_inputs(data_x0,data_x1,cleaning_split=0.05,random_seed=random_seed)

    # load samples into carl-torch format
    loading = Loader_edb()
    x, y, x0, x1 = loading.loading(
        x0 = data_x0,
        x1 = data_x1,
        save = True,
        folder = data_out_path,
        randomize = False,
        random_seed = random_seed,
        val_frac = 0.25,
        filter_outliers = True
    )
    print("Loaded new datasets ")

# now the carl-torch part 
estimator = RatioEstimator(
    n_hidden=n_hidden,
    activation="relu"
)


# pop event number, as this should not be used for training
train_loss,val_loss=estimator.train(
    method='carl',
    batch_size = 4096,
    n_epochs = n_epochs,
    x=x,
    y=y,
    x0=x0, 
    x1=x1,
    scale_inputs = True,
    #early_stopping = True,
    #early_stopping_patience = 10
)


loss_out_path=model_out_path+"/loss"
directory = os.path.dirname(loss_out_path)
if not os.path.exists(loss_out_path):
    os.makedirs(loss_out_path)

np.savetxt(loss_out_path+"/train_loss.csv", train_loss, delimiter=",")
np.savetxt(loss_out_path+"/val_loss.csv", val_loss, delimiter=",")

#
model_out_path = model_out_path +'/carl/'
print('all done, saving model to:', model_out_path)
estimator.save(model_out_path, x=x, export_model = True)

# also store patch, if it was provided:
if (args.patch!=""):
    patch_out_path=model_out_path+"/patch/"
    if not os.path.exists(patch_out_path):
        os.makedirs(patch_out_path)
    copy2(args.patch,patch_out_path)
