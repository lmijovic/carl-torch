import os
import sys
import logging
from ml import RatioEstimator
from ml.utils.loading_edb import Loader_edb
from ml.calibration import CalibratedClassifier
from ml.base import Estimator


loading = Loader_edb()
logger = logging.getLogger(__name__)
if os.path.exists("data/X_train.npy"):
    logger.info(' Doing calibration of model ')
else:
    logger.info(" No datasets available for calibration of model ")
    logger.info("ABORTING")
    sys.exit()

carl = RatioEstimator()
carl.load('model/carl/')
#load
evaluate = ['train']
X  = 'data/X_train.npy'
y  = 'data/y_train.npy'
r_hat, s_hat = carl.evaluate(X)
calib = CalibratedClassifier(carl)
calib.fit(X = X,y = y)
p0, p1, r_cal = calib.predict(X = X)
w_cal = 1/r_cal

loading.load_calibration(y_true = y,
                         p1_raw = s_hat, 
                         p1_cal = p1, 
                         plot = True,
)

evaluate = ['train']
for i in evaluate:
    p0, p1, r_cal = calib.predict(X = 'data/X0_'+i+'.npy')
    w = 1./r_cal
    loading.load_result(x0='data/X0_'+i+'.npy',
                        x1='data/X1_'+i+'.npy',
                        weights=w, 
                        label = i+'_calib', 
                        plot = True
    )

