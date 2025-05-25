"""File to plot pressure profile data"""

# import requirements #
import numpy as np
import matplotlib.pyplot as plt

# import own "modules" #
from functions_data import *
from functions_plotting import *

# import data types #
from numpy import int32, float64, ndarray

##################
# function block #
##################

def print_analysis(data:ndarray) -> None:
    """print analysis of profile (flow and weight integral)

    Args:
        data (ndarray): data array
    """
    print(f"\t flow integral: {data[6,-1]:4.1f}")
    print(f"\t mass integral: {data[5,-1]:4.1f}")

########
# main #
########
if __name__ == '__main__':

    # init #
    data_folder = "data"
    # dataset_list = ["profile_1.csv", "profile_2.csv", "profile_3.csv", "profile_4.csv", "profile_5.csv",
    #                 "shot-data-1.json", "shot-data-2.json", "shot-data-4.json", "shot-data-5.json",
    #                 "shot-data-6.json", "shot-data-7.json", "shot-data-9.json", "shot-data-10.json",
    #                 "shot-data-12.json", "shot-data-13.json", "shot-data-15.json"]
    # dataset_list = ["shot-data-1.json", "shot-data-4.json", "shot-data-7.json",
    #                 "shot-data-10.json", "shot-data-13.json"]
    dataset_list = ["shot-data-230.json", "shot-data-245.json", "shot-data-248.json", "shot-data-250.json"]
    # dataset_list = ["shot-data-700.json", "shot-data-616.json", "shot-data-617.json", "shot-data-620.json",
    #                 "shot-data-647.json", "shot-data-685.json", "shot-data-690.json"]

    # plot #
    # np.seterr(divide='ignore')
    for dataset_item in dataset_list:
        dataset, title = load_file(data_folder + "\\" + dataset_item)
        correct_temperature(dataset, temp_off=10.0)
        print('------------------------')
        print('\t', title)
        plot_hydraulic_data(dataset, title)
        # plot_temperature_data(dataset, title, temp_off=10.0)
        plt.ylim(0,12)
        # print_analysis(dataset)

    # finish #
    plt.show()
