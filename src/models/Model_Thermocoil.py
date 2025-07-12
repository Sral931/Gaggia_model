"""abstract base class for models"""
# import requirements
import numpy as np

# import own modules
from abcs.Model import Model

# import data types
from numpy import int32, float64, ndarray

class Model_Thermocoil(Model):
    # properties
    NAME: str = "Thermocoil"
    LIST_STATES: list = [
        'heater',
        'wall',
        'tube',
        'water',
        'sensor',
        'ambient'
    ]
    LIST_INPUTS: list = [
        'heater',
        'flow'
    ]

    def __init__(self, num: int = 4) -> None:
        # list states
        self.LIST_STATES = self.LIST_STATES[0:4]*num + self.LIST_STATES[-2:]

        # setup model
        super().__init__()
        self.num: int = num
        num_inputs: int = self.num_inputs

        ##############
        # capacities #
        ##############
        # capacities for heater, wall, tube+wall and water
        inv_c_base: ndarray = np.array([
            1/10.0, 1/150.0, 1/150.0, 1/17.0
        ])
        self._inv_caps: ndarray = np.zeros(4*num+2+num_inputs)
        for i in range(num):
            self._inv_caps[4*i:4*i+4] = inv_c_base[:]*num

        # sensor capacity
        self._inv_caps[4*num] = 1/20.0e-3

        ###################
        # heat conduction #
        ###################
        #      he1   wl1   tu1   wa1 
        h_base: ndarray = np.array([
            [  0.0, 10.0,  0.0,  0.0],
            [  0.0,  0.0,300.0,  0.0],
            [  0.0,  0.0,  0.0,350.0],
            [  0.0,  0.0,  0.0,  0.0]
        ])

        h_transfer: ndarray = np.array([
            [  0.0,  0.0,  0.0,  0.0],
            [  0.0,  2.7,  0.0,  0.0],
            [  0.0,  0.0,  2.7,  0.0],
            [  0.0,  0.0,  0.0,  0.0]
        ])

        h_input: ndarray = np.array([
            1.0, 0.0, 0.0, 0.0
        ])

        # put solid conduction on the diagonal
        self._heat_conduction: ndarray = np.zeros([4*num+2+num_inputs, 4*num+2+num_inputs])
        for i in range(num):
            self._heat_conduction[4*i:4*i+4, 4*i:4*i+4] = h_base[:, :]/num
            self._heat_conduction[4*i:4*i+4, -2] = h_input/num

        # put solid connections along the thermoblock on the off-diagonal
        for i in range(num-1):
            self._heat_conduction[4*i:4*i+4, 4*(i+1):4*(i+1)+4] = h_transfer/2*num

        # sensor connection to last element's tube
        self._heat_conduction[4*num, 4*num-2] = 2*4.8e-3
        
        # loss to ambient on the block elements
        for i in range(num):
            self._heat_conduction[i*4 + 1, -3] = 0.25/num
            self._heat_conduction[i*4 + 2, -3] = 0.25/num

        # symmetric part
        self._heat_conduction += self._heat_conduction.T

        # add diagonal, but correct for inputs
        self._heat_conduction -= np.diag(
            np.sum(self._heat_conduction, axis=1)
            - np.sum(self._heat_conduction[-self.num_inputs:], axis=0)
        )

        ###############
        # flow matrix #
        ###############
        self._flow_conduction: np.ndarray = np.zeros_like(self._heat_conduction)
        #      he1   wl1   tu1   wa1 
        h_flow: ndarray = np.array([
            [  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  1.0]
        ])

        # put flow mask on off diagonal
        for i in range(num-1):
            self._flow_conduction[4*i:4*i+4, 4*(i+1):4*(i+1)+4] = h_flow
        # put flow to ambient
        self._flow_conduction[4*num+1, 3] = 1.0
        
        # symmetric part
        self._flow_conduction += self._flow_conduction.T

        # add diagonal, but correct for inputs
        self._flow_conduction -= np.diag(
            np.sum(self._flow_conduction, axis=1)
            - np.sum(self._flow_conduction[-self.num_inputs:], axis=0)
        )

        # flow mask for tube-film
        self._flow_mask: ndarray = np.zeros_like(self._inv_caps)
        for i in range(num):
            self._flow_mask[4*i+3] = 1.0
    
    def jacobi(self, state:ndarray) -> ndarray:
        # build final heat conduction matrix
        heat_conduction: ndarray = np.copy(self._heat_conduction)
        
        # flow transfer
        flow: float64 = state[self.INPUT_INDEXES['flow']-1]+1e-3
        c: float64 = flow/np.pi # flow velocity in m/s
        h_flow: float64 = flow*4.196 # flow transfer in W/K

        # correct tube's film transfer
        for i in range(self.num):
            # reynolds = rho*c*d/mu, rho and d equate to 1
            # 1/mu is roughly linear with temperature
            Re: ndarray = c*1e3 * 3.0/80.0*(state[4*i]+6.7)
            # prandtl = c_p*mu/lambda
            Pr = 4.196*80.0/3.0/(state[4*i]+6.7)/0.7
            # nusselt
            Nu_lam = 0.15 * Re**0.33 * Pr**0.43
            # turbulent:
            Nu_turb = 0.021* Re**0.80 * Pr**0.43
            if Re < 2300:
                Nu = Nu_lam
            elif Re > 2900:
                Nu = Nu_turb
            else:
                Nu = (2900 - Re)/600*Nu_lam + (Re - 2300)/600*Nu_turb

            heat_conduction[4*i] *= Nu
        
        # add water flow
        heat_conduction += self._flow_conduction*h_flow
        
        # out
        return np.multiply(heat_conduction, self._inv_caps[:,None])
