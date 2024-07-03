"""Functions to simulate the boiler"""
# import requirements
import numpy as np

# import data types
from numpy import int32, float64, ndarray

class Simulator():
    NUM_MODELS: int32 = 1

    def __init__(self, index_model: int32 = 0):
        # param check
        if not (index_model in range(self.NUM_MODELS)):
            raise NotImplementedError(f'Model Index {index_model:2d} ')

        # main
        self.index_model: int32 = index_model
        self.num_states: int32 = np.shape(self.list_states())[0]

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

    def system(self) -> ndarray:
        # shortcut
        if hasattr(self, '_system'):
            result = self._system
        else:
            result = np.zeros((self.num_states, self.num_states))

            # element
            ind = 0
            result[ind] = np.array([ 0.0, 14.0, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0])
            result[ind,ind] -= np.sum(result[ind])
            # plain
            ind = 1 
            result[ind] = np.array([ 14.0, 0.0, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0.0])
            result[ind,ind] -= np.sum(result[ind])
            # water
            ind = 2
            result[ind] = np.array([ 14.7/2.0, 14.7/2.0, 0.0, 0.0, 0.0, 0.0])
            result[ind,ind] -= np.sum(result[ind])
            # group
            ind = 3
            result[ind] = np.array([ 3.6/2.0, 3.6/2.0, 0.0, 0.0, 0.0, 0.55])
            result[ind,ind] -= np.sum(result[ind])
            # body
            ind = 4
            result[ind] = np.array([ 1.8/2.0, 1.8/2.0, 0.0, 0.0, 0.0, 0.0])
            result[ind,ind] -= np.sum(result[ind])
            # ambient
            # has inf capacity due to boundary
            ind = 5
            result[ind] = np.array([ 0.0, 0.0, 0.0, 0.55, 0.0, 0.0])
            result[ind,ind] -= np.sum(result[ind])


            # save shortcut
            self._system = result
            
        # build inverse capacities
        if not hasattr(self,'_inv_caps'):
            self._inv_caps = np.array([2.0/549.0, 2.0/549.0, 1.0/422.0, 1.0/616.0, 1.0/395.0, 0.0])
        
        return np.multiply(result, self._inv_caps[:,None])

if __name__ == '__main__':
    sim = Simulator(0)
    print(sim.system())