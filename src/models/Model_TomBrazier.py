"""abstract base class for models"""
# import requirements
import numpy as np

# import own modules
from abcs.Model import Model

# import data types
from numpy import int32, float64, ndarray

class Model_TomBrazier(Model):
    """
    Model recreating Tom Brazier's results
    taken from: http://tomblog.firstsolo.net/index.php/solved-temperature-control/
    """
    # properties
    NAME: str = "TomBrazierModel"
    LIST_STATES: list = [
        'element',
        'plain',
        'sensor',
        'water',
        'group',
        'body',
        'ambient'
    ]
    LIST_INPUTS: list = [
        'heater',
        'flow'
    ]

    def __init__(self) -> None:
        # setup model
        super().__init__()
    
    def jacobi(self, state:ndarray) -> ndarray:
        # build heat conduction matrix
        if not hasattr(self, '_heat_conduction'):
            # upper heat conduction
            self._heat_conduction : ndarray = np.array([
                [0.0, 14.0, 2.4e-3, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0 , 1.0, 0.0],
                [0.0, 0.0,  0.0,    14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0 , 0.0, 0.0],
                [0.0, 0.0,  0.0,      0.0,    0.0,     0.0,     0.0 , 0.0, 0.0],
                [0.0, 0.0,  0.0,      0.0,    0.0,     0.0,     0.0 , 0.0, 0.0],
                [0.0, 0.0,  0.0,      0.0,    0.0,     0.0,     0.55, 0.0, 0.0],
                [0.0, 0.0,  0.0,      0.0,    0.0,     0.0,     0.0 , 0.0, 0.0],
                [0.0, 0.0,  0.0,      0.0,    0.0,     0.0,     0.0 , 0.0, 0.0],
                [0.0, 0.0,  0.0,      0.0,    0.0,     0.0,     0.0 , 0.0, 0.0],
                [0.0, 0.0,  0.0,      0.0,    0.0,     0.0,     0.0 , 0.0, 0.0]
            ])
            # symmetric part
            self._heat_conduction += self._heat_conduction.T
            # add diagonal, but correct for inputs
            self._heat_conduction -= np.diag(
                np.sum(self._heat_conduction, axis=1)
                - np.sum(self._heat_conduction[-self.num_inputs:], axis=0)
            )
        # build final heat conduction matrix
        heat_conduction: ndarray = np.copy(self._heat_conduction)
        

        # build inverse capacities
        if not hasattr(self,'_inv_caps'):
            self._inv_caps = np.array([2.0/549.0, 2.0/549.0, 1.0/20e-3, 1.0/422.0, 1.0/616.0, 1.0/395.0, 0.0, 0.0, 0.0])
        
        # out
        return np.multiply(heat_conduction, self._inv_caps[:,None])
