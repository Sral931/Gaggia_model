"""file that contains plotting functions"""
# import requirements
import numpy as np
import matplotlib.pyplot as plt

# import own "modules"
from functions_data import column_names, column_num, column_indexes

# import data types
from numpy import int32, float64, ndarray

###########
# DEFINES #
###########
ave = 10 # typ. timestep 0.15 to 0.2s => 10 => 1.5 to 2 s

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

def plot_hydraulic_data(data: ndarray, plt_title:str = "") -> None:
    """plot data for hydraulic evaluation

    Args:
        data (ndarray): dataset for plotting
        plt_title (str, optional): title of the dataset. Defaults to "".
    """
    # init
    fig = make_plot(-1, plt_title, 'time [s]', 'data [bar, g/s, bar*s/g]')

    # plot data
    index_data = 1 # pressure
    plt.plot(data[0], data[index_data], '.:', color='blue', label=column_names[index_data])
    plt.plot(data[0], np.convolve(data[1], np.ones(ave)/ave, mode='same'), '-', color='blue')
    plt.plot(data[0], data[2], '.:', color='yellow', label="flow")
    plt.plot(data[0], np.convolve(data[2], np.ones(ave)/ave, mode='same'), '-', color='yellow')
    plt.plot(data[0], data[5]/10, '.:', color='green', label="weight/10")
    index_data = 3 # weight flow
    plt.plot(data[0], data[index_data], '.:', color='lime', label=column_names[index_data])
    plt.plot(data[0], np.convolve(data[index_data], np.ones(ave)/ave, mode='same'), '-', color='lime')
    # index_data = 4 # temperature
    # plt.plot(data[0], data[index_data]/10, '.-', color='red', label=column_names[index_data])
    # index_data = 7 # targetTemperature
    # plt.plot(data[0], data[index_data]/10, ':', color='red', label=column_names[index_data])
    index_data = 10 # elastic flow
    plt.plot(data[0], data[index_data], '.:', color='orange', label=column_names[index_data])
    # index_data = 11 # hydr res sqrt
    # plt.plot(data[0], data[index_data], '.:', color='brown', label=column_names[index_data])
    # index_data = 12 # hydr res sqrt grad
    # plt.plot(data[0], data[index_data], '--', color='brown', label="hydr. res sqrt diff")
    # plt.plot(data[0], np.convolve(data[index_data], np.ones(ave)/ave, mode='same'), ':', color='brown')
    index_data = 11 # hydr res
    plt.plot(data[0], data[index_data], '.:', color='black', label=column_names[index_data])
    index_data = 12 # hydr res grad
    plt.plot(data[0], data[index_data], '--', color='black', label="hydr. res diff")
    # plt.plot(data[0], np.convolve(data[index_data], np.ones(ave)/ave, mode='same'), ':', color='black')
    #plt.plot(data[0], 1/data[2], '.:', color='gray', label=label+" 1/flow")
    # index_data = 14 # hydr power
    # plt.plot(data[0], data[index_data], '.:', color='black', label=column_names[index_data])

    # plot predictive reference values
    plt.hlines(
        1.1,
        plt.xlim()[0],plt.xlim()[1],color='black',linestyles='dashed')
    try:
        plt.vlines(
            data[0][int(np.where(data[11] > 0.5 )[0][0])],
            plt.ylim()[0],plt.ylim()[1],color='black',linestyles='dashed')
    except:
        pass

    # finish
    plt.ylim(-0.5,10.5)
    format_plot()

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

