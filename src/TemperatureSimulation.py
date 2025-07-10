"""file to plot temperature profiles and simulate"""
# import requirements
import numpy as np
import matplotlib.pyplot as plt

# import own modules
from models.Model_TomBrazier import *
from models.Model_Sral import *
from models.Model_Thermocoil import *
from solvers.Solver_Newton import *
from solvers.Solver_RK4 import *
from solvers.Solver_Copper import *

from functions_data import *
from functions_plotting import *

# import data types
from numpy import int32, float64, ndarray

###########
# DEFINES #
###########
temp_ambient:float64 = 20.0
FULL_HEATER_POWER: float64 = 1400

# timescale
time_min:float64 = 0.0
time_max:float64 = 5*60.0
time_points:int32 = 10*5*60
timescale:ndarray = np.linspace(time_min, time_max, time_points)
time_step:ndarray = np.zeros(time_points)
time_step[1:] = timescale[1:] - timescale[:-1]

# profile
temp_start:float64 = temp_ambient
flow:ndarray = 0.0
temp_set:ndarray = 93.0

heater:ndarray = np.zeros(time_points)

solver = Solver_RK4()
model = Model_Thermocoil(1)
# print (np.where(model._heat_conduction > 0, 1, 0))
print (model._heat_conduction)
solver.model = model

init_state: ndarray = np.ones(model.num_states)*temp_start
init_state[model.STATE_INDEXES['ambient']-1] = temp_ambient
solver.initialize(
    0.0,
    init_state,
    np.array([1400.0, 0.0])
)

########
# main #
########
for index in range(1,time_points):
    brew_delta: float64 = 0.0*(500.0 + solver.state[model.STATE_INDEXES['water']] - temp_ambient)*flow/100.0

    heater[index] = (
        FULL_HEATER_POWER if (solver.state[model.STATE_INDEXES['sensor']] < temp_set + brew_delta - 5) else 
        FULL_HEATER_POWER if (solver.state[model.STATE_INDEXES['sensor']] < temp_set + brew_delta) and (timescale[index]-np.floor(timescale[index]/3)*3 < 2.75) else 0.0
    )

    solver.solve(
        inputs=np.array( [heater[index], flow] ),
        timestep=time_step[index]
    )
    

########
# PLOT #
########
plt.figure(1)
plt.plot(solver.log[:,0], solver.log[:,1:-model.num_inputs], '--', label=model.LIST_STATES)
plt.plot(solver.log[:,0], solver.log[:,model.INPUT_INDEXES['heater']]/100, ':', label='heater/100')
plt.plot(solver.log[:,0], solver.log[:,model.INPUT_INDEXES['flow']]*10, ':', label='flow*10')

ylims = plt.ylim()
if ylims[0]< 0:
    plt.ylim(bottom = 0)
if abs(ylims[1]) > 1000:
    plt.ylim(top = 300)
format_plot()

filter_tau:float64 = 10.0
heater_smoothed: ndarray = np.mean(time_step)*np.convolve(
    heater,
    1/filter_tau*np.exp(-(timescale-time_min)/filter_tau),
    mode='full'
)[:time_points]
heater_smoothed = np.where(timescale < 40.0, np.max(heater)*np.ones(time_points), heater_smoothed)

# make_plot(title=solver.NAME+' | '+model.NAME, xlabel='time [s]', ylabel='heat transfer sum[W]')
# plt.plot(
#     timescale,
#     heater_smoothed, 
#     label='heater')
# plt.plot(
#     solver.log[:,0], 
#     (model._heat_conduction @ solver.log[:,1:].T).T,
#     label=model.LIST_STATES+model.LIST_INPUTS
# )
# format_plot()

# finish
plt.show()
