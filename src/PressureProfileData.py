import numpy as np
from json import load
import matplotlib.pyplot as plt

from ProfileDataFunctions import *

from numpy import int32, float64, ndarray

#################
# funtion block #
#################

def print_analysis(data:ndarray) -> None:
    """print analysis of profile (flow and weight integral)

    Args:
        data (ndarray): data array
    """
    print(f"\t flow integral: {data[6,-1]:4.1f}")
    print(f"\t mass integral: {data[5,-1]:4.1f}")

########
# init #
########
data_folder = "profile_data"
# dataset_list = ["profile_1.csv", "profile_2.csv", "profile_3.csv", "profile_4.csv", "profile_5.csv",
#                 "shot-data-1.json", "shot-data-2.json", "shot-data-4.json", "shot-data-5.json",
#                 "shot-data-6.json", "shot-data-7.json", "shot-data-9.json", "shot-data-10.json",
#                 "shot-data-12.json", "shot-data-13.json", "shot-data-15.json"]
dataset_list = ["shot-data-1.json", "shot-data-2.json", "shot-data-4.json", "shot-data-7.json", "shot-data-9.json", 
                "shot-data-10.json", "shot-data-12.json", "shot-data-13.json", "shot-data-15.json"]

########
# main #
########
np.seterr(divide='ignore')
for dataset_item in dataset_list:
    dataset, title = load_file(data_folder + "\\" + dataset_item)
    print('------------------------')
    print('\t', title)
    plot_hydraulic_data(dataset, title)
    # print_analysis(dataset)

plt.show()
