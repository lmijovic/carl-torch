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

# weight handling:
def crop_weight_nsig(weightarr,crop_weight_sigma):
    '''
    crop weights greater than N sigma from abs average:
    '''
    
    # checks:
    assert isinstance(weightarr,np.ndarray), print('Error: crop_weight_nperc expects ndarray, got:',type(weightarr))

    # copy to prevent overwriting original weights
    weights_ret=weightarr.copy()

    if (len(weights_ret[weights_ret < 0])>0):
        print('crop_weight_nsig: negative weights obtained, enforcing +ve weigts first.')
        ensure_positive_weight(weights_ret)

    wmax=np.mean(weights_ret)+crop_weight_sigma*np.std(weights_ret)
    print('Cropping |weights - mean| > ',crop_weight_sigma,"*sigma = ",
          crop_weight_sigma*np.std(weights_ret),"from average, cropped:",
          100.*(len(weights_ret[weights_ret>=wmax]))/len(weights_ret),"%")
    useval = np.median(weights_ret)
    weights_ret[weights_ret>=wmax]=useval

    return(weights_ret,wmax)

def crop_weight_nperc(weightarr,crop_weight_perc):
    '''
    crop large weights: crop_weight_perc fraction of the total
    '''

    # checks:
    assert isinstance(weightarr,np.ndarray), print('Error: crop_weight_nperc expects ndarray, got:',type(weightarr))

    # copy to prevent overwriting original weights
    weights_ret=weightarr.copy()

    if (len(weights_ret[weights_ret < 0])>0):
        print('crop_weight_nperc: negative weights obtained, enforcing +ve weigts first.')
        ensure_positive_weight(weights_ret)
        
    keep_weight_frac=(1-0.01*crop_weight_perc)
    print('\nCropping largest ',crop_weight_perc,'% of all weights\n')

    # find max allowed weight:
    sort_weights=weights_ret.copy()
    sort_weights.sort()
    max_elemindex=keep_weight_frac*len(sort_weights)
    if (len(sort_weights)<=max_elemindex):
        return(weights_ret,-1)

    weightmax=sort_weights[int(max_elemindex)]
    # replace weights >= max allowed weight with median
    useval = np.median(weights_ret)
    weights_ret[weights_ret>=weightmax]=useval
    return(weights_ret,weightmax)


def force_nonzero(weightarr,epsilon_div):
    '''
    expects +ve weight ndarray
    '''
    # checks:
    assert isinstance(weightarr,np.ndarray), print('Error: force_nonzero expects ndarray, got:',type(weightarr))
    if (len(weightarr[weightarr < 0])>0):
        print('Force_nonzero: negative weights obtained, enforcing +ve weigts first.')
        ensure_positive_weight(weightarr)

    # copy to prevent overwriting original weights
    weights_ret=weightarr.copy()       

    n_zero_weight=len(weights_ret[weights_ret < epsilon_div])
    if (n_zero_weight>0):
        useval = np.median(weights_ret)
        print("\nZero-weight division: replace evens in case of division by <",epsilon_div)
        print("....... replacing:",100.*n_zero_weight/len(weights_ret),'% events with:',useval,'\n')
        weights_ret[weights_ret < epsilon_div]=useval
    return(weights_ret)

def ensure_positive_weight(weightarr):

    # copy to prevent overwriting original weights
    weights_ret=weightarr.copy()

    count_replaced=0
    useval=np.median(np.abs(weights_ret))
    for i in (0,len(weights_ret)-1):
        if (weights_ret[i]<0):
            weights_ret[i]=useval
            count_replaced+=1
    if (count_replaced>0):
        print('\nReplaced ',100.*count_replaced/len(weights_ret),'% -ve weights with',useval,'\n')
    return(weights_ret)
