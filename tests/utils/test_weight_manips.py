from utils_edb import crop_weight_nsig
from utils_edb import crop_weight_nperc
from utils_edb import force_nonzero
from utils_edb import ensure_positive_weight
from helpers import draw_weights

import unittest
import numpy as np

class TestWeights(unittest.TestCase):
    
    def test_ensure_positive_weight(self):
        weights=np.array([-0.1,0.,1.,2.,3.,4.,5.,20000.])
        useval=np.median(weights)
        weights_pos = ensure_positive_weight(weights)
        self.assertTrue(np.array_equal(weights_pos,
                                       np.array([useval,0.,1.,2.,3.,4.,5.,20000.])))
        
    def test_force_nonzero(self):
        weights=np.array([0.,1.,2.,3.,4.,5.,20000.])
        useval=np.median(weights)
        weights_nonzero = force_nonzero(weights,pow(10,-12))
        target=[useval,1.,2.,3.,4.,5.,20000.]
        self.assertTrue(np.array_equal(weights_nonzero,
                                       np.array(target)))

    def test_crop_weight_nperc(self):
        weights=np.array([0.,1.,2.,3.,4.,5.,20000.])
        useval=np.median(weights)
        weights_ret,weightmax = crop_weight_nperc(weights,10.)
        target=[0.,1.,2.,3.,4.,5.,useval]
        self.assertTrue(np.array_equal(weights_ret,
                                       np.array(target)))
        self.assertEqual(weightmax,20000.)

    def test_crop_weight_nsig(self):
        weights=np.array([0.,1.,2.,3.,4.,5.,20.])
        useval=np.median(weights)
        weights_ret,weightmax = crop_weight_nsig(weights,2.)
        target=[0.,1.,2.,3.,4.,5.,useval]
        max_target=np.mean(weights)+2.*np.std(weights)
        self.assertTrue(np.array_equal(weights_ret,
                                       np.array(target)))
        self.assertEqual(weightmax,max_target)


if __name__ == '__main__':
    unittest.main()
