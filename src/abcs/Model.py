"""abstract base class for models"""
# import requirements
import numpy as np

# import data types
from numpy import int32, float64, ndarray

class Model():
    # properties
    NAME: str = "ABC Model"
    LIST_STATES: list = []
    LIST_INPUTS: list = []

    def __init__(self) -> None:
        # setup model
        self.num_states: int32 = len(self.LIST_STATES)
        self.num_inputs: int32 = len(self.LIST_INPUTS)
        self.STATE_INDEXES: dict = {name: 1 + index                   for index, name in enumerate(self.LIST_STATES)}
        self.INPUT_INDEXES: dict = {name: 1 + self.num_states + index for index, name in enumerate(self.LIST_INPUTS)}

    def jacobi(self, state:ndarray) -> ndarray:
        raise NotImplementedError("jacobi() function not properly implemented !")
