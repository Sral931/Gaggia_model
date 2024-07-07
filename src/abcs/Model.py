"""abstract base class for models"""
# import requirements
import numpy as np

# import data types
from numpy import int32, float64, ndarray

class Model():

    LIST_STATES: list = []
    LIST_INPUTS: list = []

    def __init__(self) -> None:
        # setup model
        self.num_states: int32 = len(self.LIST_STATES)
        self.num_inputs: int32 = len(self.LIST_INPUTS)

    def jacobi(self, state:ndarray) -> ndarray:
        raise NotImplementedError("jacobi() function not properly implemented !")
