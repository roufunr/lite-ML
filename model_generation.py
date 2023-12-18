import tensorflow as tf
from itertools import product
import os
import pandas
import numpy as np

hidden_layer_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180]
activations = {
    'identity': tf.identity, 
    'sigmoid': tf.keras.activations.sigmoid, 
    'tanh': tf.keras.activations.tanh,
    'relu': tf.keras.activations.relu
}
solvers = ['sgd', 'adam']
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
batch_sizes = [16, 32, 64, 128]
epochs = [10, 20, 30, 40, 50]
val_splits = [0.10, 0.15, 0.20, 0.25, 0.30]

total_simple_for_each_device = 20

dataset_path = '/Users/abdurrouf/Documents/TensorflowLiteData/F/'

def create_model(hidden_layer_size, activation, solver, learning_rate, output_shape, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation=activations[activation], input_shape=input_shape))
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    if solver == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif solver == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



    # Train your model with appropriate data and hyperparameters
    # model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    # Replace X_train, y_train, X_val, y_val with your training and validation data
    
    # Evaluate or use the model as needed
    # evaluation = model.evaluate(X_test, y_test)  # Replace X_test, y_test with your test data
    # predictions = model.predict(X_test)  # Replace X_test with the data you want to make predictions on

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

def preprocess_data():
    device_mapper = get_device_mapper()
    print(device_mapper)
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

def train_models(data, device_mapper):
    X = []
    Y = []
    for device_idx in range(len(data)):
        for sample in data[device_idx]:
            X.append(sample)
            Y.append(device_idx)
    X = np.array(X)
    Y = np.array(Y)
    parameter_combinations = list(product(hidden_layer_sizes, activations, solvers, learning_rates))
    model_counter = 1
    for params in parameter_combinations:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for val_split in val_splits: 
                    hidden_layer_size, activation, solver, learning_rate = params
                    input_shape = X[0].shape  # Update this based on your actual input shape
                    model = create_model(hidden_layer_size, activation, solver, learning_rate, 1, input_shape)
                    model.fit(X, Y, epochs=epoch, batch_size=batch_size, validation_split=val_split)
                    model.save("models/tf/"+ str(model_counter))
                    print("models/tf/"+ str(model_counter) + " DONE")
                    
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    tflite_model = converter.convert()
                    with open("models/lite/" + str(model_counter) + ".tflite", "wb") as f:
                        f.write(tflite_model)
                        print("models/lite/" + str(model_counter) + " DONE")
                    
                    model_counter += 1
                    if model_counter == 10: 
                        return 0
                    
    
    



device_mapper = get_device_mapper()
data = preprocess_data()
print(np.array(data).shape)
train_models(data, device_mapper)


# Evaluate or use the model as needed
# evaluation = model.evaluate(X_test, y_test)  # Replace X_test, y_test with your test data
# predictions = model.predict(X_test)  # Replace X_test with the data you want to make predictions on
            