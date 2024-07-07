"""Newton Solver Class"""
# import requirements
import numpy as np

# import own modules
from abcs.Solver import Solver

# import data types
from numpy import int32, float64, ndarray

class Solver_Newton(Solver):
    # properties
    NAME: str = "Newton Solver"
    # model Property from abc
    
    def __init__(self, logging: bool = True):
        super().__init__(logging)

    # initialize()
    
    def solve(self, inputs: ndarray, timestep: float64) -> None:
        # check
        if not hasattr(self, 'state'):
            raise Exception('System needs to be initialized first !')
        if np.shape(inputs)[0] != self.model.num_inputs:
            raise ValueError('Incorrect length of inputs-array !')
        # init
        self.update_aux_states(timestep, inputs)

        # main
        self.state[1:] += ( self.model.jacobi(self.state[1:]) @ self.state[1:] ) * timestep

        # log
        self.update_log()
