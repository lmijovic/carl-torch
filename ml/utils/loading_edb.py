from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging
import tarfile
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from .tools import create_missing_folders, load, load_and_check
from .plotting import draw_weighted_distributions, draw_unweighted_distributions, draw_ROC, resampled_discriminator_and_roc, plot_calibration_curve
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)


class Loader_edb():
    """
    Loading of data.
    """
    def __init__(self):
        super(Loader_edb, self).__init__()
        
    def loading(
        self,
        x0 = None,
        x1 = None,
        save = True,
        folder=None,        
        randomize = False,
        random_seed = 42,
        val_frac=0.3,
        preprocessing = True
    ):
        """
        Parameters
        ----------
 
        x0 : reference dataframe

        x1 : dataframe to reweight to

        save : bool, optional
            Save training and test samples. Default value:
            True

        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.

        randomize : bool, optional
            Randomize training sample. Default value: 
            False

        random_seed : random seed to use in all randominzations & splits

        val_frac : double 
            fraction of validation sample in train -- val split

        preprocessing : bool : exclude outliers above some margin
   

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.
        y : ndarray
            Class label with shape `(n_samples, n_parameters)`. `y=0` (`1`) for events sample from the numerator
            (denominator) hypothesis. The same information is saved as a file in the given folder.
        """

        create_missing_folders([folder])
        create_missing_folders(['plots'])

        # LM: default framework loads hard-coded samples;
        # instead we take data-frames as passes by the caller
        # check whether input dataframes were passed
        if ( x0 is None or x1 is None):
            print('Cannot create inputs, empty x0 or x1:', x0, x1)
            return(x0,x1)

        # LMTODO: crashes
        if preprocessing:
            factor = 5
            x00 = len(x0)
            x10 = len(x1)
            for column in x0.columns:
                upper_lim = x0[column].mean () + x0[column].std () * factor
                upper_lim = x1[column].mean () + x1[column].std () * factor
                lower_lim = x0[column].mean () - x0[column].std () * factor
                lower_lim = x1[column].mean () - x1[column].std () * factor
                x0 = x0[(x0[column] < upper_lim) & (x0[column] > lower_lim)]
                x1 = x1[(x1[column] < upper_lim) & (x1[column] > lower_lim)]
            x0 = x0.round(decimals=2)
            x1 = x1.round(decimals=2)
            print("filtered x0 outliers: ", (x00-len(x0))/len(x0)*100, "% ")
            print("filtered x1 outliers: ", (x10-len(x1))/len(x1)*100, "% ")

        # EDB: do this part in pandas 
        y0 = pd.DataFrame(np.zeros(x0.shape[0]))
        y1 = pd.DataFrame(np.ones(x1.shape[0]))

        x0_train, x0_val,  y0_train, y0_val =  train_test_split(x0, y0, test_size=val_frac, random_state=random_seed)
        x1_train, x1_val,  y1_train, y1_val =  train_test_split(x1, y1, test_size=val_frac, random_state=random_seed)
        x_train = pd.concat([x0_train, x1_train])
        y_train = pd.concat((y0_train, y1_train))
        x_val = pd.concat([x0_val, x1_val])
        y_val = pd.concat((y0_val, y1_val))

        #---------------------------------------------------------------
        # event number handling: 
        # pop and store event number:
        en='eventnumber'
        if en in x_train:
            x_train_en = x_train.eventnumber
            x_train = x_train.drop([en], axis=1)
            x_val_en = x_val.eventnumber
            x_val = x_val.drop([en], axis=1)
            if folder is not None and save:
                np.save(folder + "/X_train_en.npy", x_train_en.to_numpy())
                np.save(folder + "/X_val_en.npy", x_val_en.to_numpy())

        if en in x0_train:
            x0_train_en = x0_train.eventnumber
            x0_train = x0_train.drop([en], axis=1)
            x0_val_en = x0_val.eventnumber
            x0_val = x0_val.drop([en], axis=1)
            if folder is not None and save:
                np.save(folder + "/X0_train_en.npy", x0_train_en.to_numpy())
                np.save(folder + "/X0_val_en.npy", x0_val_en.to_numpy())

        if en in x1_train:
            x1_train_en = x1_train.eventnumber
            x1_train = x1_train.drop([en], axis=1)
            x1_val_en = x1_val.eventnumber
            x1_val = x1_val.drop([en], axis=1)
            if folder is not None and save:
                np.save(folder + "/X1_train_en.npy", x1_train_en.to_numpy())
                np.save(folder + "/X1_val_en.npy", x1_val_en.to_numpy())
        # end of event number handling 
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        # save data
        if folder is not None and save:
            np.save(folder  + "/X_train.npy", x_train.to_numpy())
            np.save(folder  + "/y_train.npy", y_train.to_numpy())
            np.save(folder  + "/X_val.npy", x_val.to_numpy())
            np.save(folder  + "/y_val.npy", y_val.to_numpy())
            np.save(folder  + "/X0_val.npy", x0_val.to_numpy())
            np.save(folder  + "/X1_val.npy", x1_val.to_numpy())
            np.save(folder  + "/X0_train.npy", x0_train.to_numpy())
            np.save(folder  + "/X1_train.npy", x1_train.to_numpy())
            #Tar data files if training is done on GPU
            if torch.cuda.is_available():
                plot = False #don't plot on GPU...
                tar = tarfile.open("data_out.tar.gz", "w:gz")
                for name in [folder  + "/X_train.npy", 
                             folder  + "/y_train.npy",
                             folder  + "/X_val.npy",
                             folder  + "/y_val.npy",
                             folder  + "/X0_val.npy",
                             folder  + "/X1_val.npy",
                             folder  + "/X0_train.npy",
                             folder  + "/X1_train.npy"]:
                    tar.add(name)
                tar.close()
        #---------------------------------------------------------------

        return x_train.to_numpy(), y_train.to_numpy(), x0_train.to_numpy(), x1_train.to_numpy()


    def load_result(
            self,
            x0,
            x1,
            weights = None,
            label = None,
            plot = False
    ):
        """
        Parameters
        ----------
        weights : ndarray
            r_hat weights:
        Returns
        -------
        """
        # load samples
        X0 = load_and_check(x0, memmap_files_larger_than_gb=1.0)
        X1 = load_and_check(x1, memmap_files_larger_than_gb=1.0)
        weights = weights / weights.sum() * len(X1)
        draw_ROC(X0, X1, weights, label, legend="", do="", n="calibration_plotname", plot=plot)

    def load_calibration(
        self,
        y_true,
        p1_raw = None,
        p1_cal = None,
        plot = False
    ):
        """
        Parameters
        ----------
        y_true : ndarray
            true targets
        p1_raw : ndarray
            uncalibrated probabilities of the positive class
        p1_cal : ndarray
            calibrated probabilities of the positive class
        Returns
        -------
        """

        # load samples
        y_true  = load_and_check(y_true,  memmap_files_larger_than_gb=1.0)
        plot_calibration_curve(y_true, p1_raw, p1_cal, do="", var="", save=plot)                                                                                                                                                                                                                                                                   
