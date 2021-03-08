import os
import sys
import logging
import optparse
import torch
import tarfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml import RatioEstimator
# modified loader for generic dataframe
from ml import Loader_edb

#
from variables import get_variable_names
from utils_edb import preprocess_inputs

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
    # read-in csv, skip the 1st row which contains column pseudo-names 
    data_x0=pd.read_csv(infile_old,skiprows=1, header=None,names=col_names)
    data_x1=pd.read_csv(infile_new,skiprows=1, header=None,names=col_names)

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
        val_frac = 0.1,
        preprocessing = False
    )
    print("Loaded new datasets ")

# now the carl-torch part 
estimator = RatioEstimator(
    n_hidden=(10,10,10),
    activation="relu"
)


# pop event number, as this should not be used for training
estimator.train(
    method='carl',
    batch_size = 1024,
    n_epochs = 100,
    x=x,
    y=y,
    x0=x0, 
    x1=x1,
    scale_inputs = True,
#    early_stopping = True,
#    early_stopping_patience = 10
)

#
model_out_path = model_out_path +'/carl/'
print('all done, saving model to:', model_out_path)
estimator.save(model_out_path, x=x, export_model = True)

