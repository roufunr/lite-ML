import json
import os
import csv
import pandas
import numpy as np
import tensorflow as tf
import logging
import sys

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')


logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/lmp.log',
                    filemode='a')

logger = logging.getLogger(__name__)


dataset_path = home_path + '/TensorflowLiteData/F/'

def mean(data):
    return sum(data) / len(data)

def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

def percentile(data, percentile):
    size = len(data)
    sorted_data = sorted(data)
    index = (size * percentile) // 100
    return sorted_data[int(index)]

def variance(data):
    data_mean = mean(data)
    return sum((x - data_mean) ** 2 for x in data) / len(data)

def skewness(data):
    data_mean = mean(data)
    data_variance = variance(data)
    n = len(data)
    return (sum((x - data_mean) ** 3 for x in data) / n) / (data_variance ** 1.5)

def kurtosis(data):
    data_mean = mean(data)
    data_variance = variance(data)
    n = len(data)
    return (sum((x - data_mean) ** 4 for x in data) / n) / (data_variance ** 2) - 3

def write_metrics_to_csv(metrics, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])

def load_json_to_dict(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def load_data():
    # device_mapper = load_json_to_dict(home_path + "/lite-ML/device_mapper.json")
    device_mapper = {"1": 0}
    test_sample_idx = [x for x in range(16, 20 + 1)]     # for testing accuracy change range from 16 to 20 + 1
    X = []
    Y = []
    for device_id in device_mapper:
        for sample_idx in test_sample_idx:
            csv_file_name = str(device_id) + '_' + str(sample_idx) + '.csv'
            df = pandas.read_csv(dataset_path + csv_file_name)
            two_d_list = df.values.tolist()
            one_d_list = [item for sublist in two_d_list for item in sublist]
            X.append(one_d_list)
            y = [0 for i in range(len(device_mapper))]
            y[device_mapper[device_id]] = 1
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    logger.info("Loaded data shape X:" + str(X.shape) + " Y:" + str(Y.shape))
    return X, Y

@profile
def measure_cp_on_tf_model(X, Y, tf_model):
    total_data_points = len(Y)
    for i in range(total_data_points):
        tf_model.predict(np.array([X[i]]))
    
@profile     
def measure_cp_on_lite_model(X, Y, interpreter):
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]['index']
    total_data_points = len(Y)
    for i in range(total_data_points):
        interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))
        interpreter.invoke()

model_idx = sys.argv[1]

def run_exp():
    X, Y = load_data()
    tf_model = tf.keras.models.load_model(home_path + "/downloaded_models/model_"+ str(model_idx) +"/tf")
    lite_model = tf.lite.Interpreter(model_path= home_path + "/downloaded_models/model_"+ str(model_idx) +"/" +str(model_idx)+ ".tflite")
    measure_cp_on_tf_model(X, Y, tf_model)
    measure_cp_on_lite_model(X, Y, lite_model)
run_exp()