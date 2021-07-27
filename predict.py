from variables import get_variable_names
from utils_edb import preprocess_input_file
from ml import Loader_edb
from ml import RatioEstimator
from utils_edb import crop_weight_nsig
from utils_edb import crop_weight_nperc
from utils_edb import force_nonzero
from utils_edb import ensure_positive_weight

import argparse
import pandas as pd
import numpy as np
import os

DEBUG=True

# assigns weights to a sample, based on the pre-trained model 
# the input file should contain the events to be assigned a weight,
# and provided in the same format as ref.csv used at in training 

#-----------------------------------------------------------------------------
#these should be consistent with where train_edb.py writes 
model_out_path="model"

# directory to which events+carl weights should be written in .csv:
out_csv_dir="out_csv"

# prevent 0-division:
# set this to very low, as we'll also filter large weights 
zero_w_bound = np.finfo(float).eps

# crop outlier weights more than N sigma from average
crop_weight_sigma = 5.

# alternatively: crop X% of largest weight
crop_weight_perc = -1

#-----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Predict carl weights')
parser.add_argument('--infile', required=False, help='file to process, .csv format', default='ref.csv')

args = parser.parse_args()
print(args)
col_names = get_variable_names()
print(col_names)

data_x0=pd.read_csv(args.infile,skiprows=0, header=None,names=col_names)


if (DEBUG):
    print('Processing file:',args.infile)
    print('First five lines are:')
    print(data_x0.head(5))

data_x0=preprocess_input_file(data_x0)

# load sample into carl-torch format
loading = Loader_edb()
X0, Y0, X0_eventnum = loading.load_file(
    x = data_x0,
    label = 0,
    save = False)

if (X0 is None):
    print('problem when loading, obtained empty X0, exit')
    exit(1)
if ( X0.shape[0] != Y0.shape[0] ):
    print('problem when loading, #labels does not match #events, exit')
    exit(1)   
if ( X0_eventnum is not None):
    if (X0_eventnum.shape[0] != X0.shape[0]):
        print('problem when loading, #eventnumbers does not match #events, exit')
        exit(1)
else:
    # some samples won't have an eventnumber,
    # but ATLAS ones should, since we use it to propagate the weight to reco-level events
    if (DEBUG):
        print("No eventnumber found in dataset.")


# load model and evaluate weights:
carl = RatioEstimator()
if (DEBUG):
    print('Loading model from:',model_out_path)
carl.load(model_out_path+'/carl/')
r_hat, s_hat = carl.evaluate(X0)
# prevent -ve weights (should be rounding only):
r_hat = ensure_positive_weight(r_hat)

# prevent 0-division
r_hat = force_nonzero(r_hat,zero_w_bound)

weights = 1./r_hat
weights = weights / weights.sum() * len(X0)
maxweight=-1
if (crop_weight_perc>0):
    weights,_maxweight=crop_weight_nperc(weights,crop_weight_perc)
    maxweight=max(maxweight,_maxweight)

if (crop_weight_sigma>0):
    weights,_maxweight=crop_weight_nsig(weights,crop_weight_sigma)
    maxweight=max(maxweight,_maxweight)

# write out prediction results:
if (out_csv_dir != ""):
    if not os.path.exists(out_csv_dir):
        os.makedirs(out_csv_dir)
    # check whether there was an associated event number to append
    if (X0_eventnum is not None and X0_eventnum.shape[0] == X0.shape[0]):
        X0 = np.append(X0_eventnum.reshape((len(X0),1)),X0, axis=1)
        # append weights column to the reweighted sample
        # weights have .shape of a row, we reshape to a column
        X0= np.append(X0, weights.reshape((len(X0),1)), axis=1)
        csv_path=out_csv_dir+"/predictions.csv"
        if (DEBUG):
            print('Writing predictions to:',csv_path)
        np.savetxt(csv_path, X0, delimiter=",")

exit(0)
