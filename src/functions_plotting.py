"""file that contains plotting functions"""
# import requirements
import numpy as np
import matplotlib.pyplot as plt

# import own "modules"
from functions_data import *

# import data types
from numpy import int32, float64, ndarray

###########
# DEFINES #
###########
ave = 5 # typ. timestep 0.15 to 0.2s => 10 => 1.5 to 2 s

#############
# FUNCTIONS #
#############
def make_plot(fig:int32=-1, title:str = 'title', xlabel:str='xlabel', ylabel:str='ylabel') -> int32:
    """activate or create plot

    Args:
        fig (int32, optional): figure number to activate, -1 creates new figure. Defaults to -1.
        title (str, optional): title of the figure window. Defaults to 'title'.
        xlabel (str, optional): Label of the xAxis. Defaults to 'xlabel'.
        ylabel (str, optional): Label if the yAxis. Defaults to 'ylabel'.

    Returns:
        int32: actually used figure number
    """
    # activate or create
    if fig > 0:
        fig = plt.figure(fig)
    else:
        fig = plt.figure()
    # label
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return plt.gcf().number

def format_plot() -> None:
    """finish up plot (legend and grid)"""
    plt.legend()
    plt.grid(True)

def plot_data_index(data: ndarray, index: int, scale: float = 1.0, fmt:str='.:', color:str='black', ave: int=1) -> None:
    plt.plot(data[0], scale*data[index], fmt, color=color, label=column_names[index])
    if (ave > 1):
        # plt.plot(data[0], scale*rolling_average(data[index], ave), '-', color=color)
        plt.plot(data[0], scale*kalman_filter(data[index], 0.1), '-', color=color)

def plot_hydraulic_data(data: ndarray, plt_title:str = "") -> None:
    """plot data for hydraulic evaluation

    Args:
        data (ndarray): dataset for plotting
        plt_title (str, optional): title of the dataset. Defaults to "".
    """
    # init
    fig = make_plot(-1, plt_title, 'time [s]', 'data [bar, g/s, bar*s/g]')


    # plot data
    plot_data_index(data, column_index('pressure'), color='blue', ave=ave)
    plot_data_index(data, column_index('targetPressure'), color='blue', fmt="--")
    plot_data_index(data, column_index('pumpFlow'), color='yellow', ave=ave)
    plot_data_index(data, column_index('targetPumpFlow'), color='yellow', fmt="--")
    plot_data_index(data, column_index('shotWeight'), color='green', ave=ave, scale=0.1)
    plot_data_index(data, column_index('weightFlow'), color='lime', ave=ave)
    # plot_data_index(data, column_index('temperature'), color='red', fmt='.-')
    # plot_data_index(data, column_index('targetTemperature'), color='red', fmt=':')
    plot_data_index(data, column_index('elasticFlow'), color='orange', ave=ave)
    plot_data_index(data, column_index('elasticVolume'), color='brown')
    plot_data_index(data, column_index('hydraulicResistance'), color='black')
    plot_data_index(data, column_index('hydraulicResistanceDot'), color='gray')

    # plot predictive reference values
    plt.hlines(
        1.1,
        plt.xlim()[0],plt.xlim()[1],color='gray',linestyles='dashed')
    try:
        plt.vlines(
            data[0][int(np.where(data[11] > 0.5 )[0][0])],
            plt.ylim()[0],plt.ylim()[1],color='gray',linestyles='dashed')
    except:
        pass

    # finish
    plt.ylim(-0.5,10.5)
    format_plot()

# def rolling_average(data: ndarray, num_values: int) -> ndarray:
#     # extrapolate dataset by last value
#     data_tmp: ndarray = np.append(data,np.full(int(num_values/2), data[-1]))
#     data_tmp = np.append(np.full(int(num_values/2), data[0]), data_tmp)
#     return np.convolve(data_tmp, np.ones(num_values)/num_values, mode='valid')

def plot_temperature_data(data: ndarray, plt_title:str = "", temp_off:float64 = 7.0) -> None:
    """plot data for temperature evaluation

    Args:
        data (ndarray): dataset used
        plt_title (str, optional): title for the dataset. Defaults to "".
    """
    # init
    fig = make_plot(-1, plt_title, 'time [s]', 'data [°C, g/s]')

    # plot data
    name = 'temperature'
    index = column_indexes[name]
    plt.plot(data[0], data[index]+temp_off, '-', color='red', label=name)
    # plt.plot(data[0], np.gradient(data[index])*1350, ':', color='black', label='approx boiler cap')
    name = 'targetTemperature'
    index = column_indexes[name]
    plt.plot(data[0], data[index]+temp_off, ':', color='red', label=name)
    name = 'pumpFlow'
    index = column_indexes[name]
    plt.plot(data[0], data[index]*10, '-', color='yellow', label=name + '*10')
    name = 'targetPumpFlow'
    index = column_indexes[name]
    plt.plot(data[0], data[index]*10, ':', color='yellow', label=name + '*10')

    # finish
    format_plot()

