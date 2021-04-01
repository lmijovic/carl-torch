from ml import RatioEstimator
from ml.utils.loading_edb import Loader_edb
from ml.calibration import CalibratedClassifier
from ml.base import Estimator
from ml.utils.tools import load_and_check

import numpy as np
import logging
import os
import sys

#-----------------------------------------------------------------------------
# main 
#these should be consistent with where train_edb.py writes 
data_out_path="data"
model_out_path="model"

# directory to which events+carl weights should be written in .csv:
out_csv_dir="out_csv"
#-----------------------------------------------------------------------------


loading = Loader_edb()
logger = logging.getLogger(__name__)
if not os.path.exists(data_out_path+"/X_train.npy"):
    logger.info(" No datasets available for calibration of model ")
    sys.exit()

X  = data_out_path+'/X_train.npy'
y  = data_out_path+'/y_train.npy'

carl = RatioEstimator()
carl.load(model_out_path+'/carl/')
r_hat, s_hat = carl.evaluate(X)
calib = CalibratedClassifier(carl)
calib.fit(X = X,y = y)
p0, p1, r_cal = calib.predict(X = X)

# draws a calibration control plot 
loading.load_calibration(y_true = y,
                         p1_raw = s_hat, 
                         p1_cal = p1, 
                         plot = True,
)

evaluate = ['train','val']
for i in evaluate:
    p0, p1, r_cal = calib.predict(X = data_out_path+'/X0_'+i+'.npy')
    w = 1./r_cal
    loading.load_result(x0=data_out_path+'/X0_'+i+'.npy',
                        x1=data_out_path+'/X1_'+i+'.npy',
                        weights=w, 
                        label = i+'_calib', 
                        plot = True
    )

    # dump weights to csv
    csv_path=out_csv_dir+"/"+i+"_calibrated.csv"
    x0=data_out_path+'/X0_'+i+'.npy'
    x0_enumber=data_out_path+'/X0_'+i+'_en.npy'
    X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
    X0_en=np.zeros(shape=(1,1))
    if (x0_enumber!=""):
        X0_en = load_and_check(x0_enumber, memmap_files_larger_than_gb=1.0)
    X0 = np.append(X0_en.reshape((len(X0),1)),X0, axis=1)
    X0= np.append(X0, w.reshape((len(X0),1)), axis=1)
    print("Writing .csv outputs to:", csv_path)
    np.savetxt(csv_path, X0, delimiter=",")
