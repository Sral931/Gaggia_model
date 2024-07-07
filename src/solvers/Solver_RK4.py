"""Runge Kutta O4 Solver class"""
# import requirements
import numpy as np

# import own modules
from abcs.Solver import Solver

# import data types
from numpy import int32, float64, ndarray

class Solver_RK4(Solver):
    # properties
    NAME: str = "RK4 Solver"
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
        unitary: ndarray = np.diag(np.ones(self.model.num_states+self.model.num_inputs))

        # print(unitary_mod)

        # main
        k1:ndarray = self.model.jacobi(self.state[1:])
        k2:ndarray = k1 @ (unitary + k1 * 0.5 * timestep)
        k3:ndarray = k1 @ (unitary + k2 * 0.5 * timestep)
        k4:ndarray = k1 @ (unitary + k3 * timestep)
        self.state[1:] += (k1 + 2.0*k2 + 2.0*k3 + k4) @ self.state[1:] * timestep / 6.0
        
        # log
        self.update_log()
