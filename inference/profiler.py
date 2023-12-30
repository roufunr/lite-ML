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

