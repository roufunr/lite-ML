import sys
from memory_profiler import profile as memory_profile
import os
import pandas
import numpy as np
import tensorflow as tf
import sys

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
dataset_path = home_path + '/TensorflowLiteData/F/'

model_idx = sys.argv[1]

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
    return X, Y

@memory_profile
def measure_memory_on_tf_model():
    tf_model = tf.keras.models.load_model(home_path + "/downloaded_models/model_"+ str(model_idx) +"/tf")
    X, Y = load_data()
    total_data_points = 5
    result = []
    for i in range(total_data_points):
        tf_model.predict(np.array([X[i]]))

measure_memory_on_tf_model()