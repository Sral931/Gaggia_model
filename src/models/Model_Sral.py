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
        'water1',
        'water2',
        'water3',
        'water4',
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
            #   heat   elem  plain  sens    w1     w2     w3    w4   wgr   group   body   amb  heat  flow
            self._heat_conduction : ndarray = np.array([
                [00.0,  4.0,  0.0,  0.0,    2.0,   0.0,  0.0,  0.0,  0.0,  0.0,     0.0,  0.0 ,  1.0,  0.0], # heat
                [ 0.0, 00.0, 11.0,4.8e-3,11.0*0.5, 0.0,  0.0,  0.0,  0.0,  0.1/2.0, 0.0,  0.0 ,  0.0,  0.0], # elem
                [ 0.0,  0.0, 00.0,  0.0, 11.0*0.5, 0.0,  0.0,  0.0,  0.0,  0.1/2.0, 4.8,  0.0 ,  0.0,  0.0], # plain
                [ 0.0,  0.0,  0.0, 00.0,    0.0,   0.0,  0.0,  0.0,  0.0,  0.0,     0.0,  0.0 ,  0.0,  0.0], # sens
                [ 0.0,  0.0,  0.0,  0.0,   00.0,   5.0,  0.0,  0.0,  0.0,  0.0,     0.0,  0.0 ,  0.0,  0.0], # w1
                [ 0.0,  0.0,  0.0,  0.0,    0.0,  00.0,  5.0,  0.0,  0.0,  0.0,     0.0,  0.0 ,  0.0,  0.0], # w2
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0, 00.0,  5.0,  0.0,  0.0,     0.0,  0.0 ,  0.0,  0.0], # w3
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0,  0.0, 00.0,  5.0,  3.0,     0.0,  0.0 ,  0.0,  0.0], # w4
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0,  0.0,  0.0, 00.0,  5.0,     0.0,  0.0 ,  0.0,  0.0], # wgr
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0,  0.0,  0.0,  0.0, 00.0,     0.0,  0.55,  0.0,  0.0], # group
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0,  0.0,  0.0,  0.0,  0.0,    00.0,  0.0 ,  0.0,  0.0], # body
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0,  0.0,  0.0,  0.0,  0.0,     0.0, 00.0 ,  0.0,  0.0], # amb
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0,  0.0,  0.0,  0.0,  0.0,     0.0,  0.0 , 00.0,  0.0], # heat
                [ 0.0,  0.0,  0.0,  0.0,    0.0,   0.0,  0.0,  0.0,  0.0,  0.0,     0.0,  0.0 ,  0.0, 00.0]  # flow
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
        
        # flow transfer
        h1: float64 = state[self.INPUT_INDEXES['flow']-1]*4.196
        h2: float64 = (
            state[self.STATE_INDEXES['element']-1]
            + state[self.STATE_INDEXES['plain']-1] 
            - 2*state[self.STATE_INDEXES['water4']-1])*1.0
        h3: float64 = h1 + h2

        #   heat   elem  plain  sens  w1    w2    w3    w4    wgr  group  body  amb  heat  flow
        heat_conduction += np.array([
            [00.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # heat
            [ 0.0,-h2/2,  0.0,  0.0,  0.0,  0.0,  0.0, h2/2,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # elem
            [ 0.0,  0.0,-h2/2,  0.0,  0.0,  0.0,  0.0, h2/2,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # plain
            [ 0.0,  0.0,  0.0, 00.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # sens
            [ 0.0, h2/2, h2/2,  0.0,  -h1,   h1,  0.0,  -h2,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # w1
            [ 0.0,  0.0,  0.0,  0.0,   h2,  -h3,   h1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # w2
            [ 0.0,  0.0,  0.0,  0.0,  0.0,   h2,  -h3,   h1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0], # w3
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   h2,-h2-h3,  h1,   h2,  0.0,  0.0,  0.0,  0.0], # w4
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  -h1,  0.0,  0.0,   h1,  0.0,  0.0], # wgr
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   h2,  0.0,  -h2,  0.0,  0.0,  0.0,  0.0], # group
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 00.0,  0.0,  0.0,  0.0], # body
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 00.0,  0.0,  0.0], # amb
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 00.0,  0.0], # heat
            [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 00.0]  # flow
        ])

        # build inverse capacities
        if not hasattr(self,'_inv_caps'):
            self._inv_caps = np.array([1.0/40.0, 2.0/500.0/0.6, 2.0/500.0/1.4, 1.0/20e-3, 
                                       1.0/422.0/0.2, 1.0/422/0.2, 1.0/422/0.2, 1.0/422/0.2, 1.0/422/0.2, 
                                       1.0/600.0, 1.0/100.0, 0.0, 0.0, 0.0])
        
        # out
        return np.multiply(heat_conduction, self._inv_caps[:,None])
