import tensorflow as tf
from itertools import product

hidden_layer_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180]
activations = [tf.identity, tf.keras.activations.sigmoid, tf.keras.activations.tanh, tf.keras.activations.relu]
solvers = ['sgd', 'adam']
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
warm_starts = [True, False]

parameter_combinations = list(product(hidden_layer_sizes, activations, solvers, learning_rates, warm_starts))

def create_model(hidden_layer_size, activation, solver, learning_rate, warm_start, output_shape, input_shape):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation=activation, input_shape=(input_shape,)))
    
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    if solver == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True if warm_start else False)
    elif solver == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



for i, params in enumerate(parameter_combinations):
    hidden_layer_size, activation, solver, learning_rate, warm_start = params
    print(f"Creating model {i + 1} with parameters: Hidden Layer Size: {hidden_layer_size}, Activation: {activation}, Solver: {solver}, Learning Rate: {learning_rate}, Warm Start: {warm_start}")
    model = create_model(hidden_layer_size, activation, solver, learning_rate, warm_start, 20, 276)
    
    # Train your model with appropriate data and hyperparameters
    # model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    # Replace X_train, y_train, X_val, y_val with your training and validation data
    
    # Evaluate or use the model as needed
    # evaluation = model.evaluate(X_test, y_test)  # Replace X_test, y_test with your test data
    # predictions = model.predict(X_test)  # Replace X_test with the data you want to make predictions on
