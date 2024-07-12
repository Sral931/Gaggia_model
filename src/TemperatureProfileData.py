"""file to plot temperature profiles and simulate"""
# import requirements
import numpy as np
import matplotlib.pyplot as plt

# import own modules
from models.Model_TomBrazier import *
from models.Model_Sral import *
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
temp_off:float64 = 10.0
temp_tank:float64 = 20.0
temp_ambient:float64 = 20.0
FULL_HEATER_POWER: float64 = 1300

index_dataset = 1
dataset, title = load_dataset(index_dataset, temp_off)
print(f'{index_dataset:2d} {title:20s}')
# 1 - heat up
# 2 - idle 7min after start
# 3 - temperature with flow
# 4 - temperature with high flow
# 5 - heat up group before mod
# 6 - heat up group after mod
# 7 - BoilerSideCurves
# 8 - boiler side vs brew
# 9 - temperature sensor group head

# timescale
time_min:float64 = dataset[0][0]
time_max:float64 = dataset[0][-1]
time_points:int32 = np.shape(dataset)[1]
timescale:ndarray = dataset[0]
time_step:ndarray = np.zeros(time_points)
time_step[1:] = timescale[1:] - timescale[:-1]

# profile
temp_start:float64 = dataset[column_indexes['temperature']][0] + temp_off
flow:ndarray = dataset[column_indexes['pumpFlow']]
temp_set:ndarray = dataset[column_indexes['targetTemperature']] + temp_off
temp_exp:ndarray = dataset[column_indexes['temperature']] + temp_off

heater:ndarray = np.zeros(time_points)

index_list_heaterdata = [5,6,7,8]
if index_dataset in index_list_heaterdata:
    heater = dataset[column_num+0]

solver = Solver_RK4()
model = Model_Sral()
np.set_printoptions(precision=1, suppress=True)
model.jacobi(np.zeros(model.num_states+model.num_inputs))
solver.model = model

init_state: ndarray = np.ones(model.num_states)*temp_start
init_state[model.STATE_INDEXES['group']-1] = 0.85*temp_start + 0.15*temp_ambient
init_state[model.STATE_INDEXES['body']-1] = 0.4*temp_start + 0.6*temp_ambient
init_state[model.STATE_INDEXES['ambient']-1] = temp_ambient
solver.initialize(
    0.0,
    init_state,
    np.array([1350.0, 0.0])
)

########
# main #
########
for index in range(1,time_points):
    brew_delta: float64 = (500.0 + temp_set[index] - temp_off - temp_tank)*flow[index]/100.0

    if index_dataset not in index_list_heaterdata:
        heater[index] = (
            FULL_HEATER_POWER if (solver.state[model.STATE_INDEXES['sensor']] < temp_set[index] + brew_delta - 5) else 
            FULL_HEATER_POWER if (solver.state[model.STATE_INDEXES['sensor']] < temp_set[index] + brew_delta) and (timescale[index]-np.floor(timescale[index]/3)*3 < 2.75) else 0.0
        )
        # heater[index] = (
        #     FULL_HEATER_POWER if (temp_exp[index-1] < temp_set[index] + brew_delta - 5) else 
        #     FULL_HEATER_POWER if (temp_exp[index-1] < temp_set[index] + brew_delta) and (timescale[index]-np.floor(timescale[index]/3)*3 < 2.75) else 0.0
        # )

    solver.solve(
        inputs=np.array( [heater[index], flow[index]] ),
        timestep=time_step[index]
    )
    

########
# PLOT #
########
# plot dataset temperatures
plot_temperature_data(dataset, title, temp_off=temp_off)
# plt.plot(timescale, time_step, ':', color='black')

# plot extra temperature data for certain datasets
if index_dataset in [7, 8]:
    plt.plot(timescale, dataset[column_num+1], '-', label='bottom')
    plt.plot(timescale, dataset[column_num+2], '-', label='top')

# add simulated curves to plot
# plt.figure(1)
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
