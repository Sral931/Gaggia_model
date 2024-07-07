"""Copper's SVD Solver class"""
# import requirements
import numpy as np

# import own modules
from abcs.Solver import Solver

# import data types
from numpy import int32, float64, ndarray

class Solver_Copper(Solver):
    # properties
    NAME: str = 'Copper SVD "Solver"'
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
        jacobi: ndarray = self.model.jacobi(self.state[1:])
        self.λ, self.V = np.linalg.eig(
            jacobi[:self.model.num_states, :self.model.num_states]
        )
        self.iV = np.linalg.inv(self.V)
        self.Λ, self.iΛ = np.diag(self.λ), np.diag(1.0/self.λ)
        # self.W = np.concatenate(([1.0], np.zeros(self.model.num_states-1)))

        self.ua = self.iV @ self.state[1:-self.model.num_inputs]
        self.ub = self.iV @ jacobi[:self.model.num_states,-1] * self.state[1+self.model.num_states]
        # last row contains heater reaction vector
        # self.state[1 + self.model.num_states] contains heater power

        L = np.diag(np.exp(self.λ*timestep))
        δ = np.identity(len(self.λ))
        u = L @ self.ua + np.nan_to_num(self.iΛ @ (L - δ) @ self.ub)
        self.state[1:-self.model.num_inputs] = self.V @ u

        # log
        self.update_log()
