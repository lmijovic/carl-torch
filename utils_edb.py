import pandas as pd
import numpy as np

def preprocess_inputs(df0_in,df1_in,cleaning_split=0.05,random_seed=42):
    # framework has a bunch of requirements on inputs;
    # ensure these are satisfied 
    # df0_in and df1_in = signal and background inputs 
    # cleaning_split : fraction of sample to use to ensure all carl-torch
    # requirements are satisfied.  Should be comparable/smaller
    # to your tain/val split, so that each of train and val
    # is likely to satisfy requirements individually

    # normalize weights to 1 in each sample:
    weight_sum_new = df0_in['weight'].sum()
    weight_sum_old = df1_in['weight'].sum()
    df0_in['weight']=df0_in['weight']*df0_in.shape[0]/weight_sum_new
    df1_in['weight']=df1_in['weight']*df1_in.shape[0]/weight_sum_old


    # * no NaN-s
    df0_in=df0_in.fillna(-999.)
    df1_in=df1_in.fillna(-999.)
    _df_all = pd.concat([df0_in,df1_in])
    # randomize & get a sub-sample for cleaning:
    _df_all_p1,_df_all_p2  = \
    np.split(_df_all.sample(frac=1, random_state=random_seed),
             [int(cleaning_split*len(_df_all))])

    # * no columns where all elements are the same (min == max)
    # identify column names where max is not min
    # use fraction of events = cleaning_split
    _maxs = _df_all_p1.max(axis=0)
    _mins = _df_all_p1.min(axis=0)
    _comp=_maxs.compare(_mins)

    # filter input samples to keep only these columns:
    return(df0_in[_comp.index],df1_in[_comp.index])
