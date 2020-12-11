import os
import sys
import optparse
import numpy as np

from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.utils.tools import load_and_check
from ml.utils.plotting import draw_ROC

def load_result_lm(
        x0,
        x1,
        weights = None,
        plotname = ""):
    print(np.where(np.isnan(weights)))
    X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
    X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
    #print('ewights',weights)
    #exit(0)
    weights = weights / weights.sum() * len(X1)
    draw_ROC(X0, X1, weights, label="roc",legend="",do="",n=plotname,plot=True)

#-----------------------------------------------------------------------------
# main 
#TODO Matt: set paths to model and data as in train_lm
data_out_path="data"
model_out_path="model"
# mkdir plots (for roc outputs)
#-----------------------------------------------------------------------------

carl = RatioEstimator()
carl.load(model_out_path+'/carl/')
#evaluate = ['train','val']
evaluate = ['train','val']
for i in evaluate:
    r_hat, _ = carl.evaluate(x=data_out_path + '/X0_'+i+'.npy')
    w = 1./r_hat
    load_result_lm(x0=data_out_path + '/X0_'+i+'.npy',     
                   x1=data_out_path + '/X1_'+i+'.npy',
                   weights=w,
                   plotname=i)
#carl.evaluate_performance(x=data_out_path + '/X_val.npy',y= data_out_path +'/y_val.npy')
