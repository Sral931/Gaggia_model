"""abstract base class for models"""
# import requirements
import numpy as np

# import own modules
from abcs.Model import Model

# import data types
from numpy import int32, float64, ndarray

class Model_Thermocoil(Model):
    # properties
    NAME: str = "Thermocoil"
    LIST_STATES: list = [
        'heater',
        'wall',
        'tube',
        'water',
        'sensor',
        'ambient'
    ]
    LIST_INPUTS: list = [
        'heater',
        'flow'
    ]

    def __init__(self, num: int = 4) -> None:
        # list states
        self.LIST_STATES = self.LIST_STATES[0:4]*num + self.LIST_STATES[-2:]

        # setup model
        super().__init__()

        ###################
        # heat conduction #
        ###################
        #      he1   wl1   tu1   wa1 
        h_base: ndarray = np.array([
            [  0.0,  5.0,  0.0,  0.0],
            [  0.0,  0.0,300.0,  0.0],
            [  0.0,  0.0,  0.0,350.0],
            [  0.0,  0.0,  0.0,  0.0]
        ])

        h_transfer: ndarray = np.array([
            [  0.0,  0.0,  0.0,  0.0],
            [  0.0,  2.7,  0.0,  0.0],
            [  0.0,  0.0,  2.7,  0.0],
            [  0.0,  0.0,  0.0,  0.0]
        ])

        h_input: ndarray = np.array([
            1.0, 0.0, 0.0, 0.0
        ])

        num_inputs: int = self.num_inputs
        self._heat_conduction: ndarray = np.zeros([4*num+2+num_inputs, 4*num+2+num_inputs])
        for i in range(num):
            self._heat_conduction[4*i:4*i+4, 4*i:4*i+4] = h_base[:, :]/num
            self._heat_conduction[4*i:4*i+4, -2] = h_input/num

        for i in range(num-1):
            self._heat_conduction[4*i:4*i+4, 4*(i+1):4*(i+1)+4] = h_transfer/2*num

        # sensor connection
        self._heat_conduction[4*num, 4*num-2] = 4.8e-3
        
        # loss to ambient
        for i in range(num):
            self._heat_conduction[i*4 + 1, -3] = 0.025/num
            self._heat_conduction[i*4 + 2, -3] = 0.025/num

        # symmetric part
        self._heat_conduction += self._heat_conduction.T

        # add diagonal, but correct for inputs
        self._heat_conduction -= np.diag(
            np.sum(self._heat_conduction, axis=1)
            - np.sum(self._heat_conduction[-self.num_inputs:], axis=0)
        )

        ##############
        # capacities #
        ##############
        inv_c_base: ndarray = np.array([
            1/40.0, 1/150.0, 1/150.0, 1/17.0
        ])
        self._inv_caps: ndarray = np.zeros(4*num+2+num_inputs)
        for i in range(num):
            self._inv_caps[4*i:4*i+4] = inv_c_base[:]*num

        print(self._inv_caps)
    
    def jacobi(self, state:ndarray) -> ndarray:
        # build final heat conduction matrix
        heat_conduction: ndarray = np.copy(self._heat_conduction)
        
        flow: float64 = state[self.INPUT_INDEXES['flow']-1]*4.196
        
        # out
        return np.multiply(heat_conduction, self._inv_caps[:,None])