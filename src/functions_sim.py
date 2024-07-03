"""Functions to simulate the boiler"""
# import requirements
import numpy as np

# import data types
from numpy import int32, float64, ndarray

class Simulator():
    NUM_MODELS: int32 = 1

    def __init__(self, index_model: int32 = 0, index_solver: int32 = 0):
        
        # param check
        if not (index_model in range(self.NUM_MODELS)):
            raise NotImplementedError(f'Model Index {index_model:2d} ')

        # set contruct params
        self.index_model: int32 = index_model
        self.index_solver: int32 = index_solver
        self.logging = True

        # setup model
        self.num_states: int32 = np.shape(self.list_states())[0]
        self.num_inputs: int32 = np.shape(self.list_inputs())[0]

    def list_states(self) -> list:
        match self.index_model:
            case 0:
                return [
                    'element',
                    'plain',
                    'water',
                    'group',
                    'body',
                    'ambient'
                ]

    def list_inputs(self) -> list:
        match self.index_model:
            case 0:
                return[
                    'heater'
                ]

    def system(self) -> ndarray:
        # shortcut
        if hasattr(self, '_system'):
            result = self._system
        else:
            result = np.zeros((
                self.num_states, 
                self.num_states + self.num_inputs)
            )

            # element
            ind = 0
            result[ind] = np.array([ 0.0, 14.0, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0, 1.0])
            result[ind,ind] -= np.sum(result[ind][:self.num_states])
            # plain
            ind = 1 
            result[ind] = np.array([ 14.0, 0.0, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0, 0.0])
            result[ind,ind] -= np.sum(result[ind][:self.num_states])
            # water
            ind = 2
            result[ind] = np.array([ 14.7/2.0, 14.7/2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            result[ind,ind] -= np.sum(result[ind][:self.num_states])
            # group
            ind = 3
            result[ind] = np.array([ 3.6/2.0, 3.6/2.0, 0.0, 0.0, 0.0, 0.55, 0.0])
            result[ind,ind] -= np.sum(result[ind][:self.num_states])
            # body
            ind = 4
            result[ind] = np.array([ 1.8/2.0, 1.8/2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            result[ind,ind] -= np.sum(result[ind][:self.num_states])
            # ambient
            # has inf capacity, since it's considered a steady boundary
            ind = 5
            result[ind] = np.array([ 0.0, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0])
            result[ind,ind] -= np.sum(result[ind])


            # save shortcut
            self._system = result
            
        # build inverse capacities
        if not hasattr(self,'_inv_caps'):
            self._inv_caps = np.array([2.0/549.0, 2.0/549.0, 1.0/422.0, 1.0/616.0, 1.0/395.0, 0.0])
        
        return np.multiply(result, self._inv_caps[:,None])
    
    def initialize(self, init_state:ndarray) -> None:
        self.state = init_state
        self.log = np.array([init_state])

    def solve(self, inputs: ndarray, timestep: float64) -> None:
        # check
        if not hasattr(self, 'state'):
            raise Exception('System needs to be initialized first !')
        if np.shape(inputs)[0] != self.num_inputs:
            raise ValueError('Incorrect length of inputs-array !')
        
        # main
        self.state += np.matmul(self.system(), np.append(self.state, inputs)) * timestep
        if self.logging:
            self.log = np.append(self.log, [self.state], axis=0)

if __name__ == '__main__':
    sim = Simulator(0, 0)
    sim.initialize(init_state=20.0*np.ones(sim.num_states))
    # print(sim.state)
    for index in range(60*10):
        sim.solve(1350.0*np.ones(1), 0.1)
        # print(sim.state)

    # print(sim.log[:,0])
    