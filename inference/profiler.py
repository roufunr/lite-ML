import sys
from memory_profiler import profile as memory_profile
import cProfile
from line_profiler import LineProfiler
import os
import pandas
import numpy as np
import tensorflow as tf
import logging
import sys
import csv
import time

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
dataset_path = home_path + '/TensorflowLiteData/F/'
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/profiler.log',
                    filemode='a')
logger = logging.getLogger(__name__)

model_idx = sys.argv[1]
tf_model = tf.keras.models.load_model(home_path + "/downloaded_models/model_"+ str(model_idx) +"/tf")
interpreter = tf.lite.Interpreter(model_path= home_path + "/downloaded_models/model_"+ str(model_idx) +"/" +str(model_idx)+ ".tflite")
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

X, Y = load_data()

def profile_line(func, *args, **kwargs):
    profiler = LineProfiler()
    profiler.add_function(func)
    profiler.runcall(func, *args, **kwargs)
    profiler.print_stats()

def measure_on_tf_model():
    total_data_points = len(Y)
    for i in range(total_data_points):
        tf_model.predict(np.array([X[i]]))
  
def measure_on_lite_model():
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]['index']
    total_data_points = len(Y)
    for i in range(total_data_points):
        interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))
        interpreter.invoke()

def main():
    profile_line(measure_on_tf_model)
    profile_line(measure_on_lite_model)

    with cProfile.Profile() as pr:
        measure_on_tf_model()
    pr.print_stats('cumulative')
    
    with cProfile.Profile() as pr:
        measure_on_lite_model()
    pr.print_stats('cumulative')

main()

@memory_profile
def measure_memory_on_tf_model():
    total_data_points = len(Y)
    for i in range(total_data_points):
        tf_model.predict(np.array([X[i]]))

@memory_profile
def measure_memory_on_lite_model():
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]['index']
    total_data_points = len(Y)
    for i in range(total_data_points):
        interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))
        interpreter.invoke()

measure_memory_on_tf_model()
measure_memory_on_lite_model()

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

def measure_inference_time_on_tf_model():
    total_data_points = len(Y)
    elapsed_time = []
    for i in range(total_data_points):
        start_time = time.time() * 1000
        tf_model.predict(np.array([X[i]]))
        end_time = time.time() * 1000
        elapsed_time.append(end_time - start_time)
    
    inference_times = elapsed_time
    metrics = {
        "average": mean(inference_times),
        "median": median(inference_times),
        "min": min(inference_times),
        "max": max(inference_times),
        "25th percentile": percentile(inference_times, 25),
        "75th percentile": percentile(inference_times, 75),
        "variance": variance(inference_times),
        "range": max(inference_times) - min(inference_times),
        "IQR": percentile(inference_times, 75) - percentile(inference_times, 25),
        "skewness": skewness(inference_times),
        "kurtosis": kurtosis(inference_times)
    }
    return metrics
        
def measure_inference_time_on_lite_model():
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]['index']
    total_data_points = len(Y)
    elapsed_time = []
    for i in range(total_data_points):
        interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))
        start_time = time.time() * 1000
        interpreter.invoke()
        end_time = time.time() * 1000
        elapsed_time.append(end_time - start_time)
    
    inference_times = elapsed_time
    metrics = {
        "average": mean(inference_times),
        "median": median(inference_times),
        "min": min(inference_times),
        "max": max(inference_times),
        "25th percentile": percentile(inference_times, 25),
        "75th percentile": percentile(inference_times, 75),
        "variance": variance(inference_times),
        "range": max(inference_times) - min(inference_times),
        "IQR": percentile(inference_times, 75) - percentile(inference_times, 25),
        "skewness": skewness(inference_times),
        "kurtosis": kurtosis(inference_times)
    }
    return metrics

def run_infer_time_exp():
    tf_time = measure_inference_time_on_tf_model()
    lite_time = measure_inference_time_on_lite_model()
    write_metrics_to_csv(tf_time, home_path + "/resource_utilization/" + model_idx + "/tf_time.csv" )
    write_metrics_to_csv(lite_time, home_path + "/resource_utilization/" + model_idx + "/lite_time.csv")

run_infer_time_exp()
