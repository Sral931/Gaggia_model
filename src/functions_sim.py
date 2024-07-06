"""Functions to simulate the boiler"""
# import requirements
import numpy as np
import matplotlib.pyplot as plt

# import own modules
from models.Model_TomBrazier import *
from solvers.Solver_Newton import *

# import data types
from numpy import int32, float64, ndarray

if __name__ == '__main__':
    solver = Solver_Newton()
    solver.model = Model_TomBrazier()
    solver.initialize(
        np.array([20.0]*6),
        np.array([1350.0])
    )

    time_start: float64 = 0.0
    time_steps: int32 = 360*10
    time_end: float64 = time_steps/10.0

    for index in range(time_steps):
        solver.solve(inputs=(
            2000.0 if index*0.1 < 40.0 else 
            1600.0 if index*0.1 < 90 else 
            30.0)*np.ones(1), 
            timestep=0.1
        )

    timescale: ndarray = np.linspace(time_start ,time_end, time_steps+1)
    for index in range(solver.model.num_states):
        plt.plot(timescale, solver.log[:,index], label=solver.model.LIST_STATES[index])
    for index in range(solver.model.num_inputs):
        plt.plot(timescale, solver.log[:,solver.model.num_states+index], label=solver.model.LIST_INPUTS[index])
    plt.ylim(0,180)
    plt.grid(True)
    plt.legend()
    plt.show()
    