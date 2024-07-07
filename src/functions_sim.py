"""Functions to simulate the boiler"""
# import requirements
import numpy as np
import matplotlib.pyplot as plt

# import own modules
from models.Model_TomBrazier import *
from solvers.Solver_Newton import *
from solvers.Solver_RK4 import *
from solvers.Solver_Copper import *

# import data types
from numpy import int32, float64, ndarray

if __name__ == '__main__':
    # solver = Solver_Newton()
    # solver = Solver_RK4()
    solver = Solver_Copper()
    solver.model = Model_TomBrazier()
    solver.initialize(
        0.0,
        np.array([20.0]*6),
        np.array([1350.0])
    )

    time_start: float64 = 0.0
    time_steps: int32 = 2*360*10
    time_end: float64 = time_steps/10.0

    for index in range(time_steps):
        solver.solve(inputs=(
            2000.0 if index*0.1 < 40.0 else 
            1600.0 if index*0.1 < 90 else 
            49.5)*np.ones(1), 
            timestep=0.1
        )

    timescale: ndarray = np.linspace(time_start ,time_end, time_steps+1)
    plt.figure()
    plt.title(solver.NAME+' | '+solver.model.NAME)
    plt.xlabel('time [s]')
    plt.ylabel('temperature [°C]')
    for index in range(solver.model.num_states):
        plt.plot(solver.log[:,0], solver.log[:,1+index], label=solver.model.LIST_STATES[index])
    for index in range(solver.model.num_inputs):
        plt.plot(solver.log[:,0], solver.log[:,1+solver.model.num_states+index], label=solver.model.LIST_INPUTS[index])
    plt.ylim(0,180)
    plt.grid(True)
    plt.legend()

    print(f'Heater Energy: {np.sum(solver.log[:,-1])*0.1*1e-3:7.3f} kJ')

    # plt.figure()
    # plt.title(solver.NAME+' | '+solver.model.NAME)
    # plt.xlabel('time [s]')
    # plt.ylabel('Total Heat Flow [W]')
    # plt.plot(
    #     solver.log[:,0], 
    #     (solver.model._heat_conduction @ solver.log[:,1:].T).T,
    #     label=solver.model.LIST_STATES+solver.model.LIST_INPUTS
    # )
    # plt.grid(True)
    # plt.legend()

    print(f'Heat Loss: {np.sum(solver.log[:,4]-20.0)*0.55*0.1*1e-3:7.3f} kJ')
    # print(solver.model._heat_conduction)

    plt.figure()
    plt.title(solver.NAME+' | '+solver.model.NAME)
    plt.xlabel('time [s]')
    plt.ylabel('Total Heat Energy [kJ]')
    plt.plot(
        solver.log[:,0], 
        1e-3*(np.array([549.0, 549.0, 422.0, 616.0, 395.0, 0.0, 0.0])[:,None] * solver.log[:,1:].T).T,
        label=solver.model.LIST_STATES+solver.model.LIST_INPUTS
    )
    plt.plot(
        solver.log[:,0],
        1e-3*np.sum((np.array([549.0/2.0, 549.0/2.0, 422.0, 616.0, 395.0, 0.0, 0.0])[:,None] * solver.log[:,1:].T).T, axis=1),
        label='total'
    )
    plt.grid(True)
    plt.legend()

    energy_heat_start = 1e-3*np.sum((np.array([549.0/2.0, 549.0/2.0, 422.0, 616.0, 395.0, 0.0, 0.0])[:,None] * solver.log[:,1:].T).T, axis=1)[0]
    energy_heat_end = 1e-3*np.sum((np.array([549.0/2.0, 549.0/2.0, 422.0, 616.0, 395.0, 0.0, 0.0])[:,None] * solver.log[:,1:].T).T, axis=1)[-1]
    print(f'Heat Energy Total: {energy_heat_end-energy_heat_start:7.3f} kJ')

    plt.show()
    