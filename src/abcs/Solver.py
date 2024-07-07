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

    def initialize(self, init_time:float64, init_state:ndarray, init_inputs:ndarray) -> None:
        # input check
        if np.shape(init_state)[0] != self.model.num_states:
            raise ValueError(f'Expected init_state of length {self.model.num_states:3d}, got: {np.shape(init_state)[0]:3d}')
        if np.shape(init_inputs)[0] != self.model.num_inputs:
            raise ValueError(f'Expected init_state of length {self.model.num_states:3d}, got: {np.shape(init_state)[0]:3d}')
        
        # main
        self.init_state = np.concatenate(([init_time], init_state, init_inputs))
        self.state: ndarray = np.copy(self.init_state)
        self.log: ndarray = np.zeros((1,np.shape(self.init_state)[0]))
        self.log[0] = self.init_state

    def solve(self, inputs: ndarray, timestep: float64) -> None:
        raise NotImplementedError('Solve function not properly implemented !')
    
    def update_aux_states(self, timestep:float64, inputs:ndarray) -> None:
        """updates the time (index 0) and inputs (indexes -self.model.num_inputs:)"""
        self.state[0] += timestep
        self.state[-self.model.num_inputs:] = inputs

    def update_log(self) -> None:
        """appends current state to log"""
        if self.logging:
            self.log = np.append(self.log, [self.state], axis=0)
