
import tensorflow as tf

hidden_layer_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180]
activations = {
    'identity': tf.identity, 
    'sigmoid': tf.keras.activations.sigmoid, 
    'tanh': tf.keras.activations.tanh,
    'relu': tf.keras.activations.relu
}
solvers = ['sgd', 'adam']
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
parameterSets = []
i = 0
for hiden_layer_size in hidden_layer_sizes:
    for activation in activations:
        for solver in solvers:
            for learning_rate in learning_rates:
                i += 1 
                print("p" + str(i),hiden_layer_size, activation, solver, learning_rate)
                    