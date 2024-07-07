"""Copper's SVD Solver class"""
# import requirements
import numpy as np

# import own modules
from abcs.Solver import Solver

# import data types
from numpy import int32, float64, ndarray

class Solver_Copper(Solver):
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
        
        # init
        self.update_aux_states(timestep, inputs)

        # main
        jacobi: ndarray = self.model.jacobi(self.state[1:])
        self.λ, self.V = np.linalg.eig(
            jacobi[:self.model.num_states, :self.model.num_states]
        )
        self.iV = np.linalg.inv(self.V)
        self.Λ, self.iΛ = np.diag(self.λ), np.diag(1.0/self.λ)
        self.W = np.concatenate(([1.0], np.zeros(self.model.num_states-1)))

        self.ua = self.iV @ self.state[1:-self.model.num_inputs]
        self.ub = self.iV @ jacobi[:self.model.num_states,-1] * self.state[1+self.model.num_states]

        L = np.diag(np.exp(self.λ*timestep))
        δ = np.identity(len(self.λ))
        u = L @ self.ua + np.nan_to_num(self.iΛ @ (L - δ) @ self.ub)
        self.state[1:-self.model.num_inputs] = self.V @ u

        # log
        self.update_log()

# class coppers_model:
#     def __init__(self, T0=None, ambient=20, watts=1350):
#         self.T0 = np.zeros(6, np.float64) + ambient if T0 is None else T0
#         self.W = np.zeros_like(self.T0)
#         self.W[0] = watts
        
#         self.heat_capacity = np.array([549/2, 549/2, 422, 616, 395, np.inf])
#         self.hc, self.ihc = np.diag(self.heat_capacity), np.diag(1/self.heat_capacity)
        
#         self.Wc = self.ihc @ self.W 
        
#         self.heat_loss = np.array([0,0,0,0.55,0])
#         self.heat_conduction = np.array([
#             [0, 14, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0   ],
#             [0,  0, 14.7/2.0, 3.6/2.0, 1.8/2.0, 0   ],
#             [0,  0,  0,       0,       0,       0   ],
#             [0,  0,  0,       0,       0,       0.55],
#             [0,  0,  0,       0,       0,       0   ],
#             [0,  0,  0,       0,       0,       0   ],
#         ])

#         self.heat_conduction += self.heat_conduction.T
#         self.H = -self.ihc @ (np.diag(np.sum(self.heat_conduction, axis=1)) - self.heat_conduction)
        
#         self.λ, self.V = np.linalg.eig(self.H)
#         self.iV = np.linalg.inv(self.V)
#         self.Λ, self.iΛ = np.diag(self.λ), np.diag(1.0/self.λ)

#         self.ua = self.iV @ self.T0
#         self.ub = self.iV @ self.Wc

#     def __call__(self, t):
#         L = np.diag(np.exp(self.λ*t))
#         δ = np.identity(len(self.λ))
#         u = L @ self.ua + np.nan_to_num(self.iΛ @ (L - δ) @ self.ub)
#         return self.V @ u