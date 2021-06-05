import pandas as pd
import numpy as np

# carl torch framework has a bunch of requirements on inputs;
# aim of this module is to ensure these requirements are satisfied 

def preprocess_input_file(df_in):
    # requirements on single file 

    # if weights exits, normalize weights to 1:
    if 'weight' in df_in:
        weight_sum = df_in['weight'].sum()
        df_in['weight']=df_in['weight']*df_in.shape[0]/weight_sum

    # * no NaN-s
    df_in=df_in.fillna(-999.)

    return(df_in)


def preprocess_inputs(df0_in,df1_in,cleaning_split=0.05,random_seed=42):
    # df0_in and df1_in = signal and background inputs 
    # cleaning_split : fraction of sample to use to ensure all carl-torch
    # requirements are satisfied.  Should be comparable/smaller
    # to your tain/val split, so that each of train and val
    # is likely to satisfy requirements individually

    # requirements on individual file: 
    df0_in = preprocess_input_file(df0_in)
    df1_in = preprocess_input_file(df1_in)

    # requirements on combination of both files 
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



