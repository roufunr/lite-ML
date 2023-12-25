import tensorflow as tf
import os
import sys
import pandas
import numpy as np
import csv
import time
from sklearn.preprocessing import MinMaxScaler
import logging


logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='example.log',  # Specify the log file
                    filemode='a') 

logger = logging.getLogger(__name__)

# hidden_layer_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180]
# number_of_hidden_layers = [2, 4, 6, 8]
# activations = {
#     'identity': tf.identity, 
#     'sigmoid': tf.keras.activations.sigmoid, 
#     'tanh': tf.keras.activations.tanh,
#     'relu': tf.keras.activations.relu
# }
# solvers = ['sgd', 'adam']
# learning_rates = [0.001, 0.01, 0.1]
# batch_sizes = [8, 16, 32, 64]
# epochs = [20, 40, 60, 80]

# hidden_layer_sizes = [60]
# number_of_hidden_layers = [2]
# activations = {
#     'sigmoid': tf.keras.activations.sigmoid
# }
# solvers = ['adam']
# learning_rates = [0.1]
# batch_sizes = [16]
# epochs = [60]


activations = {
    'identity': tf.identity, 
    'sigmoid': tf.keras.activations.sigmoid, 
    'tanh': tf.keras.activations.tanh,
    'relu': tf.keras.activations.relu
}

val_splits = [0.2]
total_simple_for_each_device = 20

dataset_path = '/home/rouf-linux/TensorflowLiteData/F/'

def create_model(hidden_layer_size, num_hidden_layers, activation, solver, learning_rate, output_shape, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation=activations[activation], input_shape=input_shape))
    for _ in range(num_hidden_layers - 1):
        model.add(tf.keras.layers.Dense(hidden_layer_size, activation=activations[activation]))
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    if solver == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif solver == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_device_mapper():
    file_names = os.listdir(dataset_path)
    devices_id = []
    for file_name in file_names:
        if file_name == '.DS_Store':
            continue
        device_id = int(file_name.split('_')[0])
        if device_id in devices_id:
            continue
        else:
            devices_id.append(device_id)
    devices_id = sorted(devices_id)
    mapper = {}
    for i in range(len(devices_id)):
        mapper[devices_id[i]] = i
    return mapper

def load_data():
    device_mapper = get_device_mapper()
    data = []
    for device_id in device_mapper:
        device_samples = []
        for i in range(1, (total_simple_for_each_device - 5) + 1):
            csv_file_name = str(device_id) + '_' + str(i) + '.csv'
            df = pandas.read_csv(dataset_path + csv_file_name)
            two_d_list = df.values.tolist()
            one_d_list = [item for sublist in two_d_list for item in sublist]
            device_samples.append(one_d_list)
        data.append(device_samples)
    return data

def preprocess_data(data):
    X = []
    Y = []
    for device_idx in range(len(data)):
        for sample in data[device_idx]:
            X.append(sample)
            y = [0 for i in range(23)]
            y[device_idx] = 1
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X,Y

def train_models(data, fromIdx, offset):
    X,Y = preprocess_data(data)
    df = pandas.read_csv("params.csv")
    model_counter = fromIdx + 1
    i = 0
    for index, row in df.loc[fromIdx:].iterrows():
        if i < offset:

            param_idx = index
            logger.info(str(model_counter) + "::: Training :::")
            
            hidden_layer_size = row['hidden_layer_size']
            number_of_hidden_layer = row['number_of_hidden_layer']
            activation = row['activation']
            solver = row['solver']
            learning_rate = row['learning_rate']
            batch_size = row['batch_size']
            epoch = row['epoch']
            
            input_shape = X[0].shape
            output_shape = Y[0].shape
            model = create_model(hidden_layer_size, number_of_hidden_layer, activation, solver, learning_rate, output_shape[0], input_shape)
            start_time = time.time() * 1000
            model.fit(X, Y, epochs=epoch, batch_size=batch_size, validation_split=val_splits[0])
            end_time = time.time() * 1000
            logger.info(str(model_counter) + "::: Elapsed Time ::: " + str(end_time - start_time))
            
            model.save("models/tf/"+ str(model_counter))
            logger.info("models/tf/"+ str(model_counter) + " DONE")

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open("models/lite/" + str(model_counter) + ".tflite", "wb") as f:
                f.write(tflite_model)
                logger.info("models/lite/" + str(model_counter) + " DONE")
            model_counter += 1
            tf.keras.backend.clear_session()
        else: 
            break
        i += 1
        

start_time = time.time()     
data = load_data()
# Get command-line arguments
fromIdx = int(sys.argv[1])
offset = int(sys.argv[2])
# fromIdx = 9699
# offset = 2
train_models(data, fromIdx, offset)
end_time = time.time()

logger.info("Total training time " + str(round((end_time - start_time)/60)) + " minutes")

            