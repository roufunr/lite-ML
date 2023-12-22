
import tensorflow as tf
from itertools import product
import csv

hidden_layer_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180]
number_of_hidden_layers = [2, 4, 6, 8]
activations = {
    'identity': tf.identity, 
    'sigmoid': tf.keras.activations.sigmoid, 
    'tanh': tf.keras.activations.tanh,
    'relu': tf.keras.activations.relu
}
solvers = ['sgd', 'adam']
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [8, 16, 32, 64]
epochs = [20, 40, 60, 80]


# hidden_layer_sizes = [60]
# number_of_hidden_layers = [2]
# activations = {
#     'sigmoid': tf.keras.activations.sigmoid
# }
# solvers = ['adam']
# learning_rates = [0.1]
# batch_sizes = [16]
# epochs = [60]

parameter_combinations = list(product(hidden_layer_sizes,number_of_hidden_layers, activations, solvers, learning_rates, batch_sizes, epochs))
csv_data = [['hidden_layer_size','number_of_hidden_layer', 'activation', 'solver', 'learning_rate', 'batch_size', 'epoch']]
for param in parameter_combinations:
    hidden_layer_size, number_of_hidden_layer, activation, solver, learning_rate, batch_size, epoch = param
    csv_data.append([hidden_layer_size, number_of_hidden_layer, activation, solver, learning_rate, batch_size, epoch])

file_path = 'params.csv'
with open(file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_data)
print(f"Parameters have been written to {file_path}")
