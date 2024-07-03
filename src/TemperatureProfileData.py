import numpy as np
from json import load
import matplotlib.pyplot as plt

from ProfileDataFunctions import *

from numpy import int32, float64, ndarray

#################
# funtion block #
#################
# most moved to ProfileDataFunctions

########
# init #
########
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
# 4 - heat up group before mod
# 5 - heat up group after mod
# 6 - BoilerSideCurves
# 7 - temeprature sensor group head
# 8 - boiler side vs brew

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

# heater temp array (half the total boiler)
temp_heater:ndarray = np.zeros(time_points)
temp_heater[0] = temp_start
# boiler temp array (other half of the total boiler)
temp_boiler:ndarray = np.zeros(time_points)
temp_boiler[0] = temp_start
temp_boiler_1 = np.copy(temp_boiler)
temp_sensor = np.copy(temp_boiler)
# water temp array
temp_water:ndarray = np.zeros(time_points)
temp_water[0] = temp_start
# group temp array
temp_group: ndarray = np.zeros(time_points)
temp_group[0] = 0.85*(temp_start-temp_ambient) + temp_ambient
# housing temp
temp_housing: ndarray = np.zeros(time_points)
temp_housing[0] = 0.4*(temp_start-temp_ambient) + temp_ambient

temp_plate = np.zeros(time_points)
temp_plate[0] = 0.7*(temp_start-temp_ambient) + temp_ambient
# flow array
# flow:ndarray = flow_value*np.heaviside(timescale, 0.5)*np.heaviside(flow_period - timescale, 0.5) # g / s
# heater array
heater:ndarray = np.zeros(time_points)

# machine params
cap_heater:float64 = 80.0 # J / K
res_heater_ambient:float64 = 1.0/0.02 # (W / K)^-1
res_heater_boiler:float64 = 1.0/40 # (W / K)^-1
# ambient transfer calculated by approx geometry
# total heater surface is about 400mm length and 10mm width
# approx alpha is 10 W / m^2 / K divided by 2 for way through chamber

cap_boiler: float64 = 500.0 # J / K
res_boiler: float64 = 1.0/20.0 # (W / K)^-1
res_boiler_group: float64 = 1.0/0.1 # (W / K)^-1
res_boiler_water: float64 = 1.0/20.0 # (W / K)^-1
res_boiler_ambient: float64 = 1/0.75
# internal resistance
# cast alu has about 90 W/m/K
# 2 * 400mm * 10mm cross section over half boiler width (15mm)
# (2 sides for 400mm of heater length with 10mm wall thickness)
# gives 48 W/K total
# ambient: total boiler loss ~ 60W
# 20W for insulated, 60W with some discord guy's kill-watt-measurement after 15 min
# total res to ambient ~ 0.75 W/K (60W/80K, set100C is 80C over ambient)
# group head loses ~33W by itself
# boiler res to ambient is double the total (half the total heat)

cap_sensor = 20e-3 # in J / K, about 0,05 gram of brass
res_sensor = 1/4/(2.4e-3) # in (W/K)^-1
# smooth brass on brass (~100 W / cm^2 / K)
# area of M4 about 4*3mm*2mm
# gives 2.4mW/K

cap_group: float64 = 500 # J / K #might be 420 without portafilter
res_group_water:float64 = 6.0*res_boiler_water
res_group_plate:float64 = 1.0/3.6 # (W/K)^-1
res_group_ambient: float64 = 1.0/0.2 # (W / K)^-1

cap_plate: float64 = 400 # J/K
res_plate_ambient: float64 = 1.0/0.4 # (W/K)^-1

cap_housing: float64 = 800 # J / K, equal to 2kg stainless steel
res_boiler_housing:float64 = 1.0/0.4 # (W/K)^-1
res_housing_ambient:float64 = 1.0/1.2 # (W/K)^-1
# resistance:
# surface of the boiler ~4*54mm*80mm +54mm*54mm @ 20W/m^2/K
# housing surface is much higher and is irrelevant
# to ambient:
# 6*200mm*200m (cube) @ 5 W/m^2/K

cap_water: float64 = 420 # J / K
cp_water: float64 = 4.196 # J / g / K

index_list_heaterdata = [4,5,6,8]
if index_dataset in index_list_heaterdata:
    heater = dataset[column_num+0]

########
# main #
########
for index in range(1,time_points):
    brew_delta: float64 = (500.0 + temp_water[index-1] - temp_tank)*flow[index]/100.0

    if index_dataset not in index_list_heaterdata:
        heater[index] = (
            FULL_HEATER_POWER if (temp_sensor[index-1] < temp_set[index] + brew_delta - 5) else 
            FULL_HEATER_POWER if (temp_sensor[index-1] < temp_set[index] + brew_delta) and (timescale[index]-np.floor(timescale[index]/3)*3 < 2.75) else 0.0
        )
        # heater[index] = (
        #     FULL_HEATER_POWER if (temp_exp[index-1] < temp_set[index] + brew_delta - 5) else 
        #     FULL_HEATER_POWER if (temp_exp[index-1] < temp_set[index] + brew_delta) and (timescale[index]-np.floor(timescale[index]/3)*3 < 2.75) else 0.0
        # )
    res_water_flow: float64 = 1.0/(flow[index]*cp_water+1e-6)
    # heater temp
    temp_heater[index] = temp_heater[index-1] + (
            heater[index]
            + (temp_boiler_1[index-1] - temp_heater[index-1])/(res_heater_boiler)
            + (temp_housing[index-1] - temp_heater[index-1])/res_heater_ambient
        )*time_step[index]/cap_heater
    #temp_heater[index] = temp_heater[index] if temp_heater[index] > temp_boiler_1[index-1] else temp_boiler_1[index-1]
    # boiler temp
    temp_boiler_1[index] = temp_boiler_1[index-1] + (
            (temp_heater[index-1] - temp_boiler_1[index-1])/(res_heater_boiler)
            + (temp_boiler[index-1] - temp_boiler_1[index-1])/(res_boiler)
            + (temp_group[index-1] - temp_boiler_1[index-1])/(res_boiler_group*2.0)
            + (temp_water[index-1] - temp_boiler_1[index-1])/(res_boiler_water*2.0)
            + (temp_housing[index-1] - temp_boiler_1[index-1])/(res_boiler_ambient*2.0)
        )*time_step[index]*1.5*2.0/cap_boiler
    #temp_boiler_1[index] = temp_boiler_1[index] if temp_boiler_1[index] > temp_boiler[index-1] else temp_boiler[index-1]
    # + gamma_boiler_group*(temp_group[index-1] - temp_boiler_1[index-1])
    temp_boiler[index] = temp_boiler[index-1] + (
            (temp_boiler_1[index-1] - temp_boiler[index-1])/(res_boiler)
            + (temp_group[index-1] - temp_boiler[index-1])/(res_boiler_group*2.0)
            + (temp_plate[index-1] - temp_boiler[index-1])/res_group_plate
            + (temp_water[index-1] - temp_boiler[index-1])/(res_boiler_water*2.0)
            + (temp_housing[index-1] - temp_boiler[index-1])/(res_boiler_ambient*2.0)
        )*time_step[index]*0.5*2.0/cap_boiler
    # + gamma_boiler_group*(temp_group[index-1] - temp_boiler[index-1])
    temp_sensor[index] = temp_sensor[index-1] + (
            (temp_boiler_1[index-1] - temp_sensor[index-1])/(res_sensor)
        )*time_step[index]/cap_sensor
    # water temp
    temp_water[index] = temp_water[index-1] + (
            (temp_boiler_1[index-1] - temp_water[index-1])/(res_boiler_water*2.0)
            + (temp_boiler[index-1] - temp_water[index-1])/(res_boiler_water*2.0)
            + (temp_group[index-1] - temp_water[index-1])/res_group_water
            + (temp_group[index-1] - temp_water[index-1])/res_water_flow
        )*time_step[index]/cap_water
    # temp_water[index] = temp_water[index-1] + (
    #         gamma_water_boiler*(temp_boiler[index-1] - temp_water[index-1])
    #         + gamma_water_boiler*(temp_group[index-1] - temp_water[index-1])
    #     )*time_step
    # group temp
    temp_group[index] = temp_group[index-1] + (
        (temp_water[index-1] - temp_group[index-1])/res_group_water
        + (temp_boiler[index-1] - temp_group[index-1])/(res_boiler_group*2.0)
        + (temp_boiler_1[index-1] - temp_group[index-1])/(res_boiler_group*2.0)
        + 0*(temp_plate[index-1] - temp_group[index-1])/res_group_plate
        + (temp_ambient - temp_group[index-1])/res_group_ambient
        + (temp_tank - temp_group[index-1])/res_water_flow
    )*time_step[index]/cap_group
    # + gamma_group_water*(temp_boiler[index-1] - temp_group[index-1])

    # housing temp
    temp_housing[index] = temp_housing[index-1] + (
        (temp_boiler[index-1] - temp_housing[index-1])/(res_boiler_ambient*2)
        + (temp_boiler_1[index-1] - temp_housing[index-1])/(res_boiler_ambient*2)
        + (temp_ambient - temp_housing[index-1])/res_housing_ambient
    )*time_step[index]/cap_housing

    temp_plate[index] = temp_plate[index-1] + (
        0*(temp_group[index-1] - temp_plate[index-1])/res_group_plate
        + (temp_boiler[index-1] - temp_plate[index-1])/res_group_plate
        + (temp_ambient - temp_plate[index-1])/res_plate_ambient
    )*time_step[index]/cap_plate

# plot dataset temperatures
plot_temperature_data(dataset, title, temp_off=temp_off)
# plt.plot(timescale, time_step, ':', color='black')

# plot extra temperature data for certain datasets
if index_dataset == 6:
    plt.plot(timescale, dataset[column_num+1], '-', label='bottom')
    plt.plot(timescale, dataset[column_num+2], '-', label='top')
if index_dataset == 8:
    plt.plot(timescale, dataset[column_num+1], '-', label='bottom')
    plt.plot(timescale, dataset[column_num+2], '-', label='top')

# add simulated curves to plot
# plt.figure(1)
plt.plot(timescale, temp_heater, '--', color='brown', label='temp heater')
plt.plot(timescale, temp_boiler_1, '--', color='orange')
plt.plot(timescale, temp_boiler, '--', color='orange', label='temp boiler')
plt.plot(timescale, temp_sensor, '--', color='gray', label='temp sensor')
plt.plot(timescale, temp_water, '--', color='blue', label='temp water')
plt.plot(timescale, temp_group, '--', color='green', label='temp group')
plt.plot(timescale, temp_housing, '--', color='black', label='temp housing')
plt.plot(timescale, temp_plate, '--', color='black', label='temp housing')
plt.plot(timescale, heater/100, '--', color='brown', label='heater/100')

ylims = plt.ylim()
if ylims[0]< 0:
    plt.ylim(bottom = 0)
if abs(ylims[1]) > 1000:
    plt.ylim(top = 300)

# print(np.sum(heater*time_step)/(time_max-time_min))
# print(np.sum(0.9*1350*np.where(dataset[column_indexes['temperature']] < dataset[column_indexes['targetTemperature']], 1.0, 0.0)*time_step)/(time_max-time_min))

# plt.hlines(temp_set[index],time_min, time_max, label='temp_set')
# plt.hlines(temp_set[index],time_min, time_max, label='temp_set+temp_off')
# plt.xlim(0,120)
format_plot()

filter_tau:float64 = 10.0
heater_smoothed: ndarray = np.mean(time_step)*np.convolve(
    heater,
    1/filter_tau*np.exp(-(timescale-time_min)/filter_tau),
    mode='full'
)[:time_points]
heater_smoothed = np.where(timescale < 40.0, np.max(heater)*np.ones(time_points), heater_smoothed)

make_plot(title='heat transfer', xlabel='time [s]', ylabel='heat [W]')
plt.plot(
    timescale,
    heater_smoothed, 
    label='heater')
plt.plot(
    timescale, 
    (temp_heater-temp_boiler_1)/res_heater_boiler, 
    label='heater-boiler')
plt.plot(
    timescale, 
    (temp_boiler_1-temp_boiler)/res_boiler, 
    label='boiler-boiler')
plt.plot(
    timescale, 
    (temp_boiler_1-temp_water)/2.0/res_boiler_water
    + (temp_boiler-temp_water)/2.0/res_boiler_water, 
    label='boiler-water')
plt.plot(
    timescale, 
    (temp_water-temp_group)/res_group_water, 
    label='water-group')
# plt.plot(timescale, cap_boiler/2*gamma_boiler_heater*(temp_boiler - temp_heater), label='heater-boiler')
# plt.plot(timescale, 
#          cap_boiler/2*gamma_boiler_ambient*(temp_ambient - temp_heater)
#          +cap_boiler/2*gamma_boiler_ambient*(temp_ambient - temp_boiler), 
#          label='total-ambient')
format_plot()

# finish
plt.show()
