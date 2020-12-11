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

def get_ctorch_samples(p_data_x0,p_data_x1,p_path="",clean_outliers=True,p_store=True):
    # returns S and B samples in format as used for carl torch processing
    # optionally writes out to disk in format used by carl torch

    # ensure we follow all carl torch framework requirements: 
    p_data_x0,p_data_x1=edb_preprocess_inputs(p_data_x0,p_data_x1,cleaning_split=0.02) 

    # now follow carl-torch/ml/utils/loading.py  Loader class does 

    # add labels
    p_data_x0['label']=0
    p_data_x1['label']=1

    # train-val split (pandas dataframes)
    x0_df_train = p_data_x0.sample(frac=0.8,random_state=42)
    x0_df_val = p_data_x0.drop(x0_df_train.index)
    x1_df_train = p_data_x1.sample(frac=0.8,random_state=42)
    x1_df_val = p_data_x1.drop(x1_df_train.index)
    x_df_train = pd.concat([x0_df_train,x1_df_train])
    x_df_val = pd.concat([x0_df_val,x1_df_val])

    # make sure train and val each satisfy preprocessing requirements:
    x_df_train = preprocess_inputs(x_df_train)
    x_df_val = preprocess_inputs(x_df_val)
    # preprocessing cleans up meaningless columns (as they can't be normalized)
    # need make sure columns in all train/val components are consistent:
    print('----------------------------------------')
    print(type(x_df_train.index))
    print(type(x_df_val.index))
    print('----------------------------------------')

    # set up numpy arrays as used by carl-torch:
    X0_train= (x0_df_train.drop(['label'], axis=1)).to_numpy()
    X1_train = (x1_df_train.drop(['label'], axis=1)).to_numpy()
    X0_val= (x0_df_val.drop(['label'], axis=1)).to_numpy()
    X1_val = (x1_df_val.drop(['label'], axis=1)).to_numpy()
    X_train = (x_df_train.drop(['label'], axis=1)).to_numpy()
    y_train = (x_df_train.label).to_numpy()    
    X_val = (x_df_val.drop(['label'], axis=1)).to_numpy()
    y_val = (x_df_val.label).to_numpy()

    # save data
    if p_store:
        np.save(p_path + "/X_train.npy", X_train)
        np.save(p_path + "/y_train.npy", y_train)
        np.save(p_path + "/X_val.npy", X_val)
        np.save(p_path + "/y_val.npy", y_val)
        np.save(p_path + "/X0_val.npy", X0_val)
        np.save(p_path + "/X1_val.npy", X1_val)
        np.save(p_path + "/X0_train.npy", X0_train)
        np.save(p_path + "/X1_train.npy", X1_train)
    
    return(X_train,y_train,X0_train,X1_train)

#-----------------------------------------------------------------------------
# main 
#TODO Matt: set input file name here
infile_old="ref.csv" # reference
infile_new="to_weight.csv" # to reweight
#TODO Matt: set to True or False depending on your sample
is_eeyy = True
# TODO Matt: set where outputs go (make sure these dirs exist)
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
    col_names = get_variable_names(is_eeyy)
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
