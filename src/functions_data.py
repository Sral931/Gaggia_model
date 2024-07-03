"""
Functions to load, plot and analyse Espresso profiles
"""
# import requirements #
import numpy as np
import pandas as pd
from json import load

# import data types #
from numpy import int32, float64, ndarray

###########
# DEFINES #
###########
column_names = [
    "timeInShot",
    "pressure",
    "pumpFlow",
    "weightFlow",
    "temperature",
    "shotWeight",
    "waterPumped",
    "targetTemperature",
    "targetPumpFlow",
    "targetPressure"]
# for pressure profile stuff: #
# 10 hydraulic resistance
# 11 der hydraulic resistance
# for odf data: #
# 10 heater power
column_num = len(column_names)
column_indexes = {name: index for index, name in enumerate(column_names)}

ave = 10 # typ. timestep 0.15 to 0.2s => 10 => 1.5 to 2 s

#############
# FUNCTIONS #
#############

def load_file(filepath: str) -> (ndarray, str):
    """load profile data from json file

    Args:
        filepath (str): full path to profileDataFile

    Raises:
        NotImplementedError: Raised upon incorrect filetype

    Returns:
        ndarray: data read from file
        str: title for the loaded file
    """
    # input check
    if not ('.json' in filepath or '.csv' in filepath):
        raise NotImplementedError('Only .json and .csv files are supported')

    # csv files
    if '.csv' in filepath:
        # read header
        header = ""
        names = ""
        with open(filepath) as file:
            line = file.readline()
            # continue on header lines
            while line[0] == '#':
                if 'time' in line:
                    names += line[1:]
                else:
                    header += line[1:]
                line = file.readline()

        if all(name in names for name in ['pressure', 'flow', 'mass']):
            # init data and variables
            data_csv = np.genfromtxt(filepath, encoding="utf8", delimiter=",").transpose()
            column_translation_list = [0, 1, 2, 5] # time, pressure, pump flow, shot weight
            num_points = len(data_csv[0])
            data_ndarray = np.zeros((column_num,num_points))
            # translate column indexes
            for index_column in range(len(data_csv)):
                data_ndarray[column_translation_list[index_column]] = data_csv[index_column]
            # complete variables
            data_ndarray[3] = np.divide( 
                np.gradient(data_ndarray[5]), 
                np.gradient(data_ndarray[0])) # weight flow = der(shot weight)
            data_ndarray[6] = np.cumsum(
                data_ndarray[2]*np.gradient(data_ndarray[0])) # der(water pumped) = pump_flow
        elif 'temperature' in names:
            # init data and variables
            data_csv = np.genfromtxt(filepath, encoding="utf8", delimiter=",").transpose()
            column_translation_list = [0, 4] # time, temperature
            num_points = len(data_csv[0])
            data_ndarray = np.zeros((column_num,num_points))
            # translate column indexes
            for index_column in range(len(data_csv)):
                data_ndarray[column_translation_list[index_column]] = data_csv[index_column]
            # early out
            return data_ndarray, header

    # json files
    if '.json' in filepath:
        # init data and variables
        with open(filepath) as file:
            data_json = load(file)
        
        header = f"#{data_json['id']:3d} {data_json['profile']['name']:20s}"
        data_json = data_json['datapoints']        
        num_points = len(data_json)
        data_ndarray = np.zeros((column_num,num_points))
        # translate to target format
        for index_column in range(column_num):
            for index_row in range(num_points):
                data_ndarray[index_column, index_row] = data_json[index_row][column_names[index_column]]
        # convert units
        data_ndarray[0] *= 1e-3 # time to seconds

    # calculate derived quantities
    # revert flow to not filtered
    # data_ndarray[2,1:] = data_ndarray[2,1:] + 1/0.35*(data_ndarray[2,1:]-data_ndarray[2,0:-1])

    # print(data_ndarray[temp_index])

    # elastic flow
    elastic_flow = -60*np.gradient(np.divide(data_ndarray[1], data_ndarray[0])) # calc
    elastic_flow = np.convolve(elastic_flow, np.ones(2*ave)/2/ave, mode='same') # filter
    data_ndarray = np.append(data_ndarray, [elastic_flow], axis=0)
    column_names.append("elastic flow model")
    # # put correction on flow value
    # for index_row in range(num_points):
    #     # add CO2 expansion / elastic flow, etc
    #     if (elastic_flow[index_row] < 1e-3):
    #         data_ndarray[2][index_row] -= 30*elastic_flow[index_row]
    
    #####
    # sqrt pressure shows least pronounced peaks -> less usefull
    #####
    # # hydraulic resistance sqrt
    # hydr_res = np.divide(
    #     np.sqrt(np.convolve(data_ndarray[1], np.ones(ave)/ave, mode='same')), 
    #     np.convolve(data_ndarray[2]+elastic_flow, np.ones(ave)/ave, mode='same')+1e-6)
    # hydr_res[hydr_res > 1e3] = 0.0 
    # data_ndarray = np.append(data_ndarray, 
    #                          [hydr_res],
    #                          axis = 0)
    # column_names.append("Hydraulic Resistance sqrt")

    # # gradient in hydraulic resistance
    # hydr_res_grad = np.divide(
    #     np.gradient(hydr_res), 
    #     np.gradient(data_ndarray[0])) # calc
    # hydr_res_grad = np.convolve(hydr_res_grad, np.ones(ave)/ave, mode='same') # filter
    # # hydr_res_grad = np.divide(hydr_res_grad, hydr_res + 1e-3)*10
    # data_ndarray = np.append(data_ndarray, 
    #                          [hydr_res_grad],
    #                          axis=0) # der hydr. res.
    # column_names.append("der. Hydraulic Resistance sqrt")

    # hydraulic resistance
    hydr_res = np.divide(
        np.convolve(data_ndarray[1], np.ones(ave)/ave, mode='same'), 
        np.convolve(data_ndarray[2], np.ones(ave)/ave, mode='same')+1e-6)
    hydr_res[hydr_res > 1e3] = 0.0 
    data_ndarray = np.append(data_ndarray, 
                             [hydr_res],
                             axis = 0)
    column_names.append("Hydraulic Resistance")

    # gradient in hydraulic resistance
    hydr_res_grad = np.divide(
        np.gradient(hydr_res), 
        np.gradient(data_ndarray[0])) # calc
    hydr_res_grad = np.convolve(hydr_res_grad, np.ones(ave)/ave, mode='same') # filter
    # hydr_res_grad = np.divide(hydr_res_grad, hydr_res + 1e-3)*10
    data_ndarray = np.append(data_ndarray, 
                             [hydr_res_grad],
                             axis=0) # der hydr. res.
    column_names.append("der. Hydraulic Resistance")

    # hydraulic power
    hydr_pow = data_ndarray[1] * data_ndarray[2]
    data_ndarray = np.append(data_ndarray, [hydr_pow], axis=0)
    column_names.append("Hydraulic Power")

    # return
    return data_ndarray, header

def correct_temperature(dataset: ndarray, temp_off:int32 = 7.0) -> None:
    """correct the read temperature if above setpoint

    Args:
        dataset (ndarray): _description_
    """
    temp_index = column_indexes['temperature']
    target_temp_index = column_indexes['targetTemperature']
    dataset[temp_index] = np.where(
    dataset[temp_index] > dataset[target_temp_index],
    (dataset[temp_index] - dataset[target_temp_index])*(dataset[target_temp_index]+temp_off),
    dataset[temp_index]
)

def load_dataset(index_dataset: int32, temp_off:float64 = 7.0) -> (ndarray, str):
    """loads pre-defined datasets and corrects them

    Args:
        index_dataset (int32): dataset index to load

    Notes:
        index 0: temperature on startup
        index 1: temperature on 50C startup with 4g/s flow

    Returns:
        ndarray: loaded and corrected dataset
        str: title of the dataset
    """
    # for index in range(1,column_num):
    #     plt.plot(dataset[0], dataset[index], label=f'{index:2d}')
    # format_plot()
    # plt.show()

    # input check
    if (index_dataset < 0):
        raise ValueError(f'Dataset index {index_dataset:2d} not allowed !')
    
    # custom flow profile
    if (index_dataset == 0):
        period = 30
        time_start: float64 = -100.0
        time_end: float64 = 240.0
        num_points = int((time_end-time_start)*10)
        
        dataset = np.zeros((column_num,num_points))
        dataset[0] = np.linspace(time_start, time_end, num_points)
        dataset[column_indexes['pumpFlow']] = np.where( (dataset[0] > 0) & (dataset[0] < period), 4.0, 0.0 )
        dataset[column_indexes['temperature']] = 90.0
        dataset[column_indexes['targetTemperature']] = 90.0
        return dataset, 'custom flow profile'

    path_data_folder = 'data\\'

    # heat up profile
    if (index_dataset == 1):
        dataset, title = load_file(path_data_folder+'shot-data-67.json')
        # corr certain values
        correct_temperature(dataset, temp_off) # temp reading above set
        return dataset, 'heat up'
    
    # idle profile about 7 min after startup
    if (index_dataset == 2):
        dataset, title = load_file(path_data_folder+'shot-data-72.json')
        # corr certain values
        correct_temperature(dataset, temp_off) # temp reading above set
        dataset[column_indexes['pumpFlow']] = 0.0 # erase pumpFlow
        return dataset, 'idle 7min after start'

    # heat up with flow
    if (index_dataset == 3):
        dataset1, title = load_file(path_data_folder+'shot-data-58.json')
        # corr certain values for combination
        dataset1[column_indexes['pumpFlow']][-1] = 0.0
        correct_temperature(dataset1, temp_off)
        dataset1[column_indexes['targetTemperature']][-1] = 1.0
        dataset2, title = load_file(path_data_folder+'shot-data-59.json')
        # corr certain values for combination
        dataset2[0] += 65.0 # time
        dataset2[column_indexes['pumpFlow']][0:10] = 0.0 # pumpFlow at start
        correct_temperature(dataset2, temp_off) # temperature reading above set
        # combine
        dataset = np.append(dataset1, dataset2, axis=1)
        return dataset, 'temperature with flow'
    
    # brew head heat profile
    if (index_dataset == 4):
        dataset, title = load_file(path_data_folder+'BeforeBrewHead.csv')
        timescale_interp = np.linspace(0, np.floor((dataset[0][-1]-dataset[0][0])*10)/10.0, int(np.floor((dataset[0][-1]-dataset[0][0])*10)))
        dataset_interp = [ 
            timescale_interp if index == 0 else np.interp(timescale_interp,dataset[0],dataset[index]) 
            for index in range(column_num)
        ]
        dataset = np.array(dataset_interp)
        # corr certain values
        dataset[column_indexes['temperature']] -= temp_off # corr temp reading
        dataset[column_indexes['targetTemperature']] = np.where(dataset[0] < 60, 110.0, 20.0)
        dataset = np.append(
            dataset, 
            [np.where((0 < dataset[0]) & (dataset[0] < 60), 1350, 0)],
            axis=0
        ) # heater power
        return dataset, 'heat up group before mod'
    
    # brew head heat profile
    if (index_dataset == 5):
        dataset, title = load_file(path_data_folder+'AfterBrewHead.csv')
        timescale_interp = np.linspace(0, np.floor((dataset[0][-1]-dataset[0][0])*10)/10.0, int(np.floor((dataset[0][-1]-dataset[0][0])*10)))
        dataset_interp = [ 
            timescale_interp if index == 0 else np.interp(timescale_interp,dataset[0],dataset[index]) 
            for index in range(column_num)
        ]
        dataset = np.array(dataset_interp)
        # corr certain values
        dataset[column_indexes['temperature']] -= temp_off # corr temp reading
        dataset[column_indexes['targetTemperature']] = np.where(dataset[0] < 60, 110.0, 20.0)
        dataset = np.append(
            dataset, 
            [np.where((0 < dataset[0]) & (dataset[0] < 60), 1350, 0)],
            axis=0
        ) # heater power
        return dataset, 'heat up group after mod'
    
    # temp sensors file front face
    if (index_dataset == 6):
        pandas_excel = pd.read_excel(path_data_folder+'Probe-points.ods', engine='odf')
        pandas_dataset = pandas_excel.to_numpy().transpose()
        column_translation_list = [0, column_num+0, column_num+1, column_num+2, 4] # time, heater, bottom, top, middle
        num_points = len(pandas_dataset[0])
        dataset = np.zeros((column_num+3,num_points))
        # translate column indexes
        for index_column in range(len(pandas_dataset)):
            dataset[column_translation_list[index_column]] = pandas_dataset[index_column]
        dataset[column_indexes['temperature']] -= temp_off # corr temp reading
        return dataset, 'BoilerSideCurves'
    
    # temp sensors group head
    if (index_dataset == 7):
        pandas_excel = pd.read_excel(path_data_folder+'Grouphead_Temperatures.xlsx')
        # print(pandas_excel.info())
        pandas_dataset = pandas_excel.to_numpy().transpose()[:2]
        column_translation_list = [0, 4] # time, grouphead temperature
        num_points = len(pandas_dataset[0])
        dataset = np.zeros((column_num,num_points))
        # translate column indexes
        for index_column in range(len(pandas_dataset)):
            dataset[column_translation_list[index_column]] = pandas_dataset[index_column]
        dataset[column_indexes['temperature']] -= temp_off # corr temp reading
        dataset[column_indexes['targetTemperature']] = np.where(dataset[0] < 980, 90.0, 93.0)
        dataset[column_indexes['pumpFlow']] = np.where((1690 < dataset[0]) & (dataset[0] < 1696), 8.0, 0.0)
        return dataset, 'temp sensor grouphead'
    
    # temp sensors file front face vs brew
    if (index_dataset == 8):
        dataset_bottom, title = load_file(path_data_folder+'BrewThermostatCurves_Bottom.csv')
        dataset_brew, title = load_file(path_data_folder+'BrewThermostatCurves_Brew.csv')
        dataset_top, title = load_file(path_data_folder+'BrewThermostatCurves_Top.csv')
        timescale_interp = np.linspace(
            0, 
            np.floor((dataset_brew[0][-1]-dataset_brew[0][0])*10)/10.0, 
            int(np.floor((dataset_brew[0][-1]-dataset_brew[0][0])*10))
        )
        dataset_interp = [ 
            timescale_interp if index == 0 else np.interp(timescale_interp,dataset_brew[0],dataset_brew[index]) 
            for index in range(column_num)
        ]
        dataset = np.array(dataset_interp)
        dataset = np.append(
            dataset, 
            [np.where((5 < dataset[0]) & (dataset[0] < 38), 2500, 0)],
            axis=0
        ) # heater power
        dataset = np.append(
            dataset, 
            [np.interp(timescale_interp,dataset_bottom[0], dataset_bottom[column_indexes['temperature']])],
            axis=0
        )
        dataset = np.append(
            dataset, 
            [np.interp(timescale_interp,dataset_top[0], dataset_top[column_indexes['temperature']])],
            axis=0
        )
        dataset[column_indexes['temperature']] -= temp_off # corr temp reading
        return dataset, 'BoilerSideCurves vs brew'
    
    raise NotImplementedError(f'Dataset index {index_dataset:2d} is not defined !')

