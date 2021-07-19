import os
import numpy as np

from sklearn import preprocessing

from ml import RatioEstimator
from ml.utils.tools import load_and_check
from ml.utils.plotting import draw_ROC
from helpers.draw_weights import draw_weights

def eval_and_store(
        x0,
        x1,
        weights,
        crop_weight_sigma=-1,
        plotname="",
        extra_text="",
        x0_enumber="",
        x1_enumber="",
        csv_path = ""):
    X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
    X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
    X0_en=np.zeros(shape=(1,1))
    X1_en=np.zeros(shape=(1,1))
    if (x0_enumber!=""):
        X0_en = load_and_check(x0_enumber, memmap_files_larger_than_gb=1.0)
    if (x1_enumber!=""):
        X1_en = load_and_check(x1_enumber, memmap_files_larger_than_gb=1.0)

    draw_weights(weights,crop_weight_sigma=crop_weight_sigma,extra_text=extra_text)


    if (crop_weight_sigma>0):
        # crop weights greater than N sigma from abs average:
        absweights=np.abs(weights)
        wmax=np.mean(absweights)+crop_weight_sigma*np.std(absweights)
        print('Cropping |weights - mean| > ',crop_weight_sigma,"*sigma from average, cropped:",
              100.*(len(weights[weights>wmax])+len(weights[weights<-1.*wmax]))/len(weights),"%")
        weights[weights>wmax]=1
        weights[weights<-1.*wmax]=1
        
    weights = weights / weights.sum() * len(X0)

 
    # ROC Curve: this is calculate with a separate ML algorithm
    # we need to scale inputs: scale together, than split them back
    Xall = np.concatenate((X0,X1), axis=0)
    # standard scaling
    #scaler = preprocessing.StandardScaler().fit(Xall)
    #Xall_scaled = scaler.transform(Xall)
    # minmax scaling
    Xall_scaled = (Xall-np.min(Xall,axis=0))/(np.max(Xall,axis=0)-np.min(Xall,axis=0))
    x0_scaled = Xall_scaled[0:X0.shape[0],:]
    x1_scaled = Xall_scaled[X0.shape[0]:,:]

    # n = plot name extra (we leave it to default)
    draw_ROC(x0_scaled, x1_scaled, weights, label="roc",legend="",do="",n=plotname,plot=True)

    if (csv_path != ""):
        # check whether there was an associated event number to append
        if (X0_en.shape[0] == X0.shape[0]):
            X0 = np.append(X0_en.reshape((len(X0),1)),X0, axis=1)
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

# stamp for plots when using ATLAS samples
extra_text="ATLAS Simulation, Work in Progress"

# prevent 0-division:
# set this to very low, as we'll also filter large weights 
zero_w_bound = np.finfo(float).eps

# crop (=set to 1) outlier weights more than N sigma from average
crop_weight_sigma = 10

#-----------------------------------------------------------------------------

if not os.path.exists(out_csv_dir):
    os.makedirs(out_csv_dir)

carl = RatioEstimator()
carl.load(model_out_path+'/carl/')

evaluate = ['train','val']
for i in evaluate:

    x0=data_out_path + '/X0_'+i+'.npy'
    r_hat, s_hat = carl.evaluate(x=data_out_path + '/X0_'+i+'.npy')
##    print('what is Carl returning?')
##    r=r_hat[0]
##    s=s_hat[0]
##    print('r=p0/p1,s=p0/(p0+p1)')
##    print(r,s,r/(1+r))
##    print('r=p1/p0,s=p0/(p0+p1)') # this 
##    print(r,s,1/(1+r))
##    print('r=p0/p1,s=p1/(p0+p1)') # this 
##    print(r,s,1/(1+r))
##    print('r=p1/p0,s=p1/(p0+p1)')
##    print(r,s,r/(1+r))
##
##    exit(0)
    # Carl is returning this:
    # s_hat = p1 / (p0+p1), where 0 corresponds to X0
    # r_hat = p0/p1
    # weight will be assigned to X0, hence has to correspond to p1/p0

    # how should we prevent 0-division?
    n_zero_weight=len(r_hat[r_hat < zero_w_bound])
    if (n_zero_weight>0):
        print("Filtering zero-weight events:",100.*n_zero_weight/len(r_hat),'%')
        r_hat[r_hat < zero_w_bound]=1.

    # now evaluate the weight to apply to X0
    w = 1./r_hat
    
    # check whether there was an associated event number path:
    x0_enumber_path = data_out_path + '/X0_'+i+'_en.npy'
    if not os.path.isfile(x0_enumber_path):
        x0_enumber_path=""
    x1_enumber_path = data_out_path + '/X1_'+i+'_en.npy'
    if not os.path.isfile(x1_enumber_path):
        x1_enumber_path=""

    # roc curve and storing
    eval_and_store(x0=data_out_path + '/X0_'+i+'.npy',     
                   x1=data_out_path + '/X1_'+i+'.npy',
                   weights=w,
                   crop_weight_sigma=crop_weight_sigma,
                   plotname=i,
                   extra_text=extra_text,
                   x0_enumber=x0_enumber_path,
                   x1_enumber=x1_enumber_path,
                   csv_path=out_csv_dir+"/"+i+".csv")

