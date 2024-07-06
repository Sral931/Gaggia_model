"""abstract base class for models"""
# import requirements
import numpy as np

# import own modules
from abcs.Model import Model

# import data types
from numpy import int32, float64, ndarray

class Model_TomBrazier(Model):

    LIST_STATES: list = [
        'element',
        'plain',
        'water',
        'group',
        'body',
        'ambient'
    ]

    LIST_INPUTS: list = [
        'heater'
    ]

    def __init__(self) -> None:
        # setup model
        super().__init__()
    
    def jacobi(self, state:ndarray) -> ndarray:
        # build heat conduction matrix
        if not hasattr(self, '_heat_conduction'):
            # upper heat conduction
            self._heat_conduction : ndarray = np.array([
                [0.0, 14.0, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0 ],
                [0.0, 0.0,  14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0 ],
                [0.0, 0.0,  0.0,      0.0,     0.0,     0.0 ],
                [0.0, 0.0,  0.0,      0.0,     0.0,     0.55],
                [0.0, 0.0,  0.0,      0.0,     0.0,     0.0 ],
                [0.0, 0.0,  0.0,      0.0,     0.0,     0.0 ],
            ])
            # symmetric part
            self._heat_conduction += self._heat_conduction.T
            # add diagonal
            self._heat_conduction -= np.diag(np.sum(self._heat_conduction, axis=1))
            # add reaction to inputs
            self._heat_conduction = np.append(
                self._heat_conduction, 
                np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                axis=0
            ).T
        heat_conduction: ndarray = np.copy(self._heat_conduction)
        

        # build inverse capacities
        if not hasattr(self,'_inv_caps'):
            self._inv_caps = np.array([2.0/549.0, 2.0/549.0, 1.0/422.0, 1.0/616.0, 1.0/395.0, 0.0])
        
        return np.multiply(heat_conduction, self._inv_caps[:,None])