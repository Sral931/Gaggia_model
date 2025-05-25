"""
Functions to run a kalman with derivative
Useful library: https://github.com/rlabbe/filterpy/tree/master/filterpy
"""
# import requirements #
import numpy as np
import pandas as pd
from json import load

# import data types #
from numpy import int32, float64, ndarray

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

def plot_data_index(data: ndarray, index: int, scale: float = 1.0, fmt:str='.:', color:str='black', label:str="label") -> None:
    plt.plot(data[0], scale*data[index], fmt, color=color, label=label)

def kalman_deriv(data: ndarray, err_model: float64, err_meas: float64) -> ndarray:
    """
    kalman filter incoming data to a model with a derivative

    Args:
        data (ndarray): data, including time and state

    Returns:
        ndarray: output of filtered data
    """
    def F(timestep) -> ndarray:
        return np.array([[1.0,timestep],[0.0,1.0]])
    def Q(timestep, q_c) -> ndarray:
        return np.array([[1.0/3.0*timestep**3, 1.0/2.0*timestep**2],[1.0/2.0*timestep**2, timestep]])*q_c

    data = np.append(data,[np.zeros_like(data[0])], axis=0)
    data = np.append(data,[np.zeros_like(data[0])], axis=0)

    x_pred: ndarray = np.array([[data[2,0]], [0.0]])
    P_pred: ndarray = np.array([[err_meas, 0.0],[0.0, err_meas]])
    H_meas: ndarray = np.array([[1.0, 0.0]])
    S_mat: ndarray = np.array([0.0])
    R_mat: ndarray = np.array([err_meas])
    K_mat: ndarray = np.array([0.0, 0.0])

    ind_t: int32 = 0 # index time
    ind_d: int32 = 2 # index data
    ind_r: int32 = -2 # result index
    timestep: float64 = 0.0
    data[ind_r, 0] = x_pred[0,0]
    data[ind_r+1, 0] = x_pred[1,0]
    for index in range(1, np.shape(data)[1]):
        # predict
        timestep = data[ind_t][index]-data[ind_t][index-1]
        F_mat = F(timestep)
        x_pred = F_mat @ x_pred
        P_pred = F_mat @ P_pred @ F_mat.T + Q(timestep, np.sqrt(err_model)) + np.array([[1.0,0.0],[0.0,0.0]])*np.abs(x_pred[:,0]-data[ind_r:, index])
        S_mat = H_meas @ P_pred @ H_meas.T + R_mat
        K_mat = P_pred @ H_meas.T @ np.linalg.inv(S_mat)

        # update
        x_delta = K_mat @ (data[ind_d,index]-H_meas @ x_pred)
        x_pred = x_pred + x_delta
        KH = K_mat @ H_meas
        P_delta = KH @ P_pred
        P_pred = P_pred - P_delta

        data[ind_r, index] = x_pred[0,0]
        data[ind_r+1, index] = x_pred[1,0]


    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_values: int32 = 512
    total_time: float64 = 10.0
    noise_level = 0.2

    column_names = ["time", "true", "measurement"]

    data = np.zeros((3,num_values))

    data[0] = np.linspace(0.0, total_time, num_values, endpoint=True)
    data[1] = np.sin(data[0]*2*2*np.pi/total_time)
    val_lim = 0.7
    data[1, np.abs(data[1]) > val_lim ] = np.sign(data[1, np.abs(data[1]) > val_lim ])*val_lim
    data[2] = data[1] + np.random.randn(num_values)*noise_level
    data = kalman_deriv(data, 0.01, noise_level**2)

    make_plot(fig=-1, title="Derivative Kalman Test", xlabel="time", ylabel="data")
    plot_data_index(data, 1, fmt="--", color="blue", label="orig")
    plot_data_index(data, 2, fmt=":", color="red", label="noisy")
    plot_data_index(data, 3, fmt="-", color="green", label="kalman o0")
    plot_data_index(data, 4, fmt="-", color="orange", label="kalman o1")
    # format_plot()

    plt.show()
