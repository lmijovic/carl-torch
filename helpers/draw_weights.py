import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from sklearn import preprocessing

plt.rcParams.update({'font.size': 16})
# activate latex text rendering
#rc('text', usetex=True)

def get_w(p_arr):
    sum_weights=float(len(p_arr))
    numerators = np.ones_like(p_arr)
    return(numerators/sum_weights)

def draw_weights(carl_weights=[],maxweight=-1,extra_text=""):
    '''
    show carl weights distribution in lin/log x-scales
    '''

    carl_weights=abs(carl_weights)

    # draw vertical line where weight outliers will be cropped
    xsig=-1
    crop_str=""
    crop_line_color='firebrick'
    if (maxweight>0):
        xsig=maxweight
        # nominal display
        crop_str='cut'
        # manually set nice display
        #crop_str='$\\frac{|w-\mathrm{mean}|}{\sigma}>$ 5'
    # linear scale 
    plt.figure()
    plt.hist(abs(carl_weights), bins=(100),
             histtype='step', linewidth=2, weights=get_w(carl_weights))
    plt.yscale('log')

    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    yspan=ymax-ymin
    xmin, xmax = axes.get_xlim()
    xspan=xmax-xmin
    if (maxweight>0):
        plt.axvline(xsig,linewidth=1.5, color=crop_line_color)
        # add info on cropped value 
        xpos=max(xsig-0.01*xspan,xmin)
        plt.text(xpos, 0.8*yspan, crop_str,
                 rotation='horizontal', verticalalignment = 'top' , horizontalalignment = 'right' )

    if (extra_text!=""):
        plt.text(xmax, ymin+1.05*yspan, extra_text, horizontalalignment = 'right',fontsize=12)

    plt.xlabel('CARL weight',horizontalalignment='right', x=1.0)
    plt.ylabel('Fraction of events',horizontalalignment='right', y=1.0)
    plt.savefig("carl_weight.png", bbox_inches='tight',dpi=200)
    plt.savefig("carl_weight.pdf", bbox_inches='tight')
   
    # log scale 
    plt.figure()
    logw=np.log10(carl_weights)
    plt.hist(logw, bins=(100),histtype='step',
             weights=get_w(carl_weights))
    plt.yscale('log')

    if (maxweight>0):
        plt.axvline(np.log10(xsig),linewidth=1.5, color=crop_line_color)
        # add info on cropped value 
        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        yspan=ymax-ymin
        xmin, xmax = axes.get_xlim()
        xspan=xmax-xmin
        xpos=max(xsig-0.1*xspan,xmin)
        if (xpos>0):
            plt.text(np.log10(xpos), 0.8*yspan, crop_str,
                     rotation='horizontal', verticalalignment = 'top' , horizontalalignment = 'right' )

    if (extra_text!=""):
        plt.text(xmax, ymin+1.05*yspan, extra_text, horizontalalignment = 'right',fontsize=12)
    plt.xlabel('log10(CARL weight)',horizontalalignment='right', x=1.0)
    plt.ylabel('Fraction of events',horizontalalignment='right', y=1.0)
    plt.savefig("carl_weight_log.png", bbox_inches='tight',dpi=200)
    plt.savefig("carl_weight_log.pdf", bbox_inches='tight')
    
