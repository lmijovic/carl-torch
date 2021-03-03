import os
import sys
import optparse
import numpy as np

from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.utils.tools import load_and_check
from ml.utils.plotting import draw_ROC

def eval_and_store(
        x0,
        x1,
        weights,
        csv_path = ""):
    X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
    X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
    print(type(weights),print(weights))
    weights = weights / weights.sum() * len(X1)
    #print('Indices of any events with NAN weights? ', np.where(np.isnan(weights)))
    # n = plot name extra (we leave it to default)
    draw_ROC(X0, X1, weights, label="roc",legend="",do="",n="",plot=True)

    if (csv_path != ""):
        # append weights column to the reweighted sample
        # weights have .shape of a row, we reshape to a column
        X0= np.append(X0, weights.reshape((len(X0),1)), axis=1)
        print("Writing .csv outputs to:", csv_path)
        np.savetxt(csv_path, X0, delimiter=",")

#-----------------------------------------------------------------------------
# main 
#these should be consistent with where train_edb.py writes 
data_out_path="data"
model_out_path="model"

# directory to which events+carl weights should be written in .csv:
out_csv_dir="out_csv"
#-----------------------------------------------------------------------------

if not os.path.exists(out_csv_dir):
    os.makedirs(out_csv_dir)

carl = RatioEstimator()
carl.load(model_out_path+'/carl/')

evaluate = ['train','val']
for i in evaluate:
    r_hat, _ = carl.evaluate(x=data_out_path + '/X0_'+i+'.npy')
    w = 1./r_hat
    eval_and_store(x0=data_out_path + '/X0_'+i+'.npy',     
                   x1=data_out_path + '/X1_'+i+'.npy',
                   weights=w,
                   csv_path=out_csv_dir+"/"+i+".csv")

