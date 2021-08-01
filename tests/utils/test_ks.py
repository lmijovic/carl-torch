from utils_edb import ks_w2
import matplotlib.pyplot as plt

import unittest
import numpy as np
from scipy import stats

class TestKS2(unittest.TestCase):
    
    def test_ksw2_trivialweight(self):
        n1=100
        rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1, random_state=42)
        rvs2 = stats.norm.rvs(size=n1, loc=0.0, scale=1, random_state=42)
        # control: scipi, unweighted
        ksout=stats.ks_2samp(rvs1, rvs2,alternative='two-sided')
        # assign weights from a histogram
        bins = np.linspace(-3,3,100)
        binsvals1=plt.hist(rvs1, bins=bins)
        binsvals2=plt.hist(rvs2, bins=bins)
        weights1=rvs1/rvs1
        weights2=rvs1/rvs2
        #print(type(weights1),binsvals1[0])
        for ind in range (0,len(rvs2)):
            # find the bin to which the value belongs:
            for ibin in range(0,len(bins)):
                if (bins[ibin]>rvs2[ind]):
                    if (0==(binsvals2[0])[ibin]):
                        weights2[ind]=1.
                    else:
                        weights2[ind]= (binsvals1[0])[ibin]/(binsvals2[0])[ibin]
                    break
        #print(weights2)

        ksval,pval=ks_w2(rvs1,rvs2,weights1, weights2,alternative='two-sided')
        print(ksout,ksval,pval)
        # the toy scenario should yield identical result:
        self.assertEqual(ksout.pvalue,pval)

    def test_ksw2_weight(self):
        n1=1000
        rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1, random_state=42)
        rvs2 = stats.norm.rvs(size=n1, loc=0.1, scale=1.2, random_state=22)
        # control: scipi, unweighted
        ksout=stats.ks_2samp(rvs1, rvs2,alternative='two-sided')
        # our weigted  
        weights1=rvs1/rvs1
        weights2=rvs1/rvs2
        # assign weights from a histogram
        bins = np.linspace(-3,3,1000)
        binsvals1=plt.hist(rvs1, bins=bins)
        binsvals2=plt.hist(rvs2, bins=bins)
        weights1=rvs1/rvs1
        weights2=rvs1/rvs2
        #print(type(weights1),binsvals1[0])
        for ind in range (0,len(rvs2)):
            # find the bin to which the value belongs:
            for ibin in range(0,len(bins)):
                if (bins[ibin]>rvs2[ind]):
                    if (0==(binsvals2[0])[ibin]):
                        weights2[ind]=1.
                    else:
                        weights2[ind]= (binsvals1[0])[ibin]/(binsvals2[0])[ibin]
                    break
        #print(weights2)      

        ksval,pval=ks_w2(rvs1,rvs2,weights1, weights2,alternative='two-sided')
        # reweighting should improve:
        print(ksout,ksval,pval)
        self.assertGreater(pval,ksout.pvalue)


if __name__ == '__main__':
    unittest.main()
