"""abstract base class for solvers"""
# import requirements
import numpy as np

# import own modules
from abcs.Solver import Solver

# import data types
from numpy import int32, float64, ndarray

class Solver_Newton(Solver):
    # model Property
    
    def __init__(self, logging: bool = True):
        super().__init__(logging)

    # initialize()
    
    def solve(self, inputs: ndarray, timestep: float64) -> None:
        # check
        if not hasattr(self, 'state'):
            raise Exception('System needs to be initialized first !')
        if np.shape(inputs)[0] != self.model.num_inputs:
            raise ValueError('Incorrect length of inputs-array !')
        
        # main
        self.state += np.matmul(self.model.jacobi(np.append(self.state, inputs)), np.append(self.state, inputs)) * timestep
        if self.logging:
            self.log = np.append(self.log, [np.append(self.state, inputs)], axis=0)
