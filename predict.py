from variables import get_variable_names
from utils_edb import preprocess_input_file
from ml import Loader_edb
from ml import RatioEstimator
from utils_edb import crop_weight_nsig
from utils_edb import crop_weight_nperc
from utils_edb import force_nonzero
from utils_edb import ensure_positive_weight
from utils_edb import ks_w2

# KS test
from scipy import stats

import argparse
import pandas as pd
import numpy as np
import os

DEBUG=True

'''
 assigns weights to a sample, based on the pre-trained model 
 the input file should contain the events to be assigned a weight,
 and provided in the same format as ref.csv used at in training 

'''

#-----------------------------------------------------------------------------
# settings:
#these should be consistent with where train_edb.py writes 
model_out_path="model"

# directory to which events+carl weights should be written in .csv:
# if empty, no predictions will be written out 
#out_csv_dir="out_csv"
out_csv_dir=""

#  KS test can be scheduled to quantify similarity of weighted vs baseline
do_ks = True
do_weighted_ks_tH = True

# prevent 0-division:
# set this to very low, as we'll also filter large weights 
zero_w_bound = np.finfo(float).eps


#-----------------------------------------------------------------------------
# params
parser = argparse.ArgumentParser(description='Predict carl weights')
parser.add_argument('--infile', required=False, help='file to process, .csv format', default='ref.csv')
parser.add_argument('--to_weight_file', required=False, help='target file to which we are weighting, used for KS test of weighting quality', default='to_weight.csv')
parser.add_argument('--crop_weight_sigma', required=False, help='crop weights > crop_weight_sigma from average', default=5)
parser.add_argument('--crop_weight_perc', required=False, help='crop largest crop_weight_perc weights', default=-1)

args = parser.parse_args()
infile=args.infile
to_weight_file=args.to_weight_file
crop_weight_sigma=float(args.crop_weight_sigma)
crop_weight_perc=float(args.crop_weight_perc)

#-----------------------------------------------------------------------------

col_names = get_variable_names()
data_x0=pd.read_csv(infile,skiprows=0, header=None,names=col_names)

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
#  ensure <weights>=1 after cropping
weights = weights*len(weights)/weights.sum()

maxweight=-1
if (crop_weight_perc>0):
    weights,_maxweight=crop_weight_nperc(weights,crop_weight_perc)
    maxweight=max(maxweight,_maxweight)

if (crop_weight_sigma>0):
    weights,_maxweight=crop_weight_nsig(weights,crop_weight_sigma)
    maxweight=max(maxweight,_maxweight)

#  ensure <weights>=1 after cropping
weights = weights*len(weights)/weights.sum()

# append weights column to the reweighted sample
# weights have .shape of a row, we reshape to a column
X0 = np.append(X0, weights.reshape((len(X0),1)), axis=1)

# write out prediction results:
if (out_csv_dir != ""):
    if not os.path.exists(out_csv_dir):
        os.makedirs(out_csv_dir)
    # check whether there was an associated event number to append
    if (X0_eventnum is not None and X0_eventnum.shape[0] == X0.shape[0]):
        X0 = np.append(X0_eventnum.reshape((len(X0),1)),X0, axis=1)
        csv_path=out_csv_dir+"/predictions.csv"
        if (DEBUG):
            print('Writing predictions to:',csv_path)
        np.savetxt(csv_path, X0, delimiter=",")

if (do_ks):


    # load 
    data_x1=pd.read_csv(to_weight_file,skiprows=0, header=None,names=col_names)

    ks_weights=np.full((0,len(col_names)), 1)

    if (do_weighted_ks_tH):
        # let's use a sub-set of observables,
        # and give larger weight to FS ones and the ones used by Tom's NN 
        col_names_filt=["t_pt","t_eta","H_pt","H_eta","b_pt","b_eta",
                   "dEta_t_H","theta_t_H","m_tH","theta_tH_b","dR_tH_b","whatever"]
        ks_weights=np.full((1,len(col_names)), 2)
        ks_weights[0,0]=1.
        ks_weights[0,1]=1.
        ks_weights[0,2]=1.
        ks_weights[0,3]=1.
 
    # do a couple of samplings of the reference data-set, to avoid picking KS stat. fluctuations
    n_x1_splits=5
    sampling_frac=0.5
    imprs=[1]*n_x1_splits

    for x1_split in range(0,n_x1_splits):
        data_x1_split=data_x1.sample(frac=sampling_frac, random_state=x1_split).reset_index(drop=True)

        ksstat_now=0
        ksstat_w=0
        index=0
        for name in col_names:
            # trivial weights:
            weights1=np.full(data_x1_split[name].shape[0], 1)
            weights0=np.full(data_x0[name].shape[0], 1)
            ksstat,ksprob=ks_w2(data_x1_split[name],data_x0[name],weights1,weights0,alternative='two-sided')
            ksstat_now+=ksstat*ks_weights[0,index]
            # actual: 
            ksstat,ksprob=ks_w2(data_x1_split[name],data_x0[name],weights1,weights,alternative='two-sided')
            ksstat_w+=ksstat*ks_weights[0,index]
            index+=1
            
            # evaluate improvement:
            impr=(ksstat_now-ksstat_w)/(ksstat_w+ksstat_now)

        imprs[x1_split]=impr

    # median or last:
    elem=min(int(n_x1_splits/2.)-1,n_x1_splits-1)
    imprs.sort
    imprs_sigma=0.5*(imprs[n_x1_splits-1]+imprs[0]-2.*imprs[elem])
    print("Relative improvement is KS statistic is : ", imprs[elem], "+-", imprs_sigma)
    print("Weight spread is:",np.log10(np.amax(weights))-np.log10(np.amin(weights)))
exit(0)
