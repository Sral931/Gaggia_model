"""abstract base class for solvers"""
# import requirements
import numpy as np

# import own modules
from abcs.Model import Model

# import data types
from numpy import int32, float64, ndarray

class Solver():
    def __init__(self, logging: bool = True):

        # set contruct params
        self.logging = True

        self.model = Model()

    def initialize(self, init_time:float64, init_state:ndarray, init_input:ndarray) -> None:
        # input check
        if np.shape(init_state)[0] != self.model.num_states:
            raise ValueError(f'Expected init_state of length {self.model.num_states:3d}, got: {np.shape(init_state)[0]:3d}')
        if np.shape(init_input)[0] != self.model.num_inputs:
            raise ValueError(f'Expected init_state of length {self.model.num_states:3d}, got: {np.shape(init_state)[0]:3d}')
        
        # main
        self.state: ndarray = init_state
        self.log: ndarray = np.zeros((1,1+self.model.num_states+self.model.num_inputs))
        self.log[0, 0] = init_time
        self.log[0, 1:1+self.model.num_states] = init_state
        self.log[0,-self.model.num_inputs:] = init_input

    def solve(self, inputs: ndarray, timestep: float64) -> None:
        raise NotImplementedError('Solve function not properly implemented !')
