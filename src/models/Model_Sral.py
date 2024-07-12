"""abstract base class for models"""
# import requirements
import numpy as np

# import own modules
from abcs.Model import Model

# import data types
from numpy import int32, float64, ndarray

class Model_Sral(Model):
    # properties
    NAME: str = "SralModel"
    LIST_STATES: list = [
        'heater',
        'element',
        'plain',
        'sensor',
        'water',
        'watergroup',
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
            #   heat   elem  plain sens     water  watgr   group   body   amb  heat  flow
            self._heat_conduction : ndarray = np.array([
                [0.0,   5.0, 0.0,  0.0,      0.0,    0.0   ,  0.0,     0.0,  0.0 , 1.0, 0.0],
                [0.0,   0.0, 20.0, 4.8e-3, 20.0*0.3, 0.0   ,  0.8/2.0, 0.0,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,    20.0*0.7, 0.0   ,  0.8/2.0, 4.8,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    0.0e-3,  0.0,     0.0,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    5.0   ,  0.0,     0.0,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    0.0   ,  5.0,     0.0,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    0.0   ,  0.0,     0.0,  0.55, 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    0.0   ,  0.0,     0.0,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    0.0   ,  0.0,     0.0,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    0.0   ,  0.0,     0.0,  0.0 , 0.0, 0.0],
                [0.0,   0.0, 0.0,  0.0,      0.0,    0.0   ,  0.0,     0.0,  0.0 , 0.0, 0.0]
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
        
        flow: float64 = state[self.INPUT_INDEXES['flow']-1]*4.196
        #   heat  elem  plain sens  water watgr group body   amb  heat  flow
        heat_conduction += np.array([
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,-flow, flow,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0, flow,-2*flow,0.0,  0.0, flow,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0, flow,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
        ])

        # build inverse capacities
        if not hasattr(self,'_inv_caps'):
            self._inv_caps = np.array([1.0/40.0, 2.0/500.0/0.7, 2.0/500.0/1.3, 1.0/20e-3, 1.0/422.0/0.8, 1.0/422/0.2, 1.0/600.0, 1.0/400.0, 0.0, 0.0, 0.0])
        
        # out
        return np.multiply(heat_conduction, self._inv_caps[:,None])
