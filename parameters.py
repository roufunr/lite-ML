hidden_layer_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180]
activations = ['identity', 'logisitc', 'tanh', 'relu']
solvers = ['lbfgs', 'sgd', 'adam']
learning_rates = ['contant', 'invscaling', 'adaptive']
warm_starts = [True, False]
parameterSets = []
i = 0
for hiden_layer_size in hidden_layer_sizes:
    for activation in activations:
        for solver in solvers:
            for learning_rate in learning_rates:
                for warm_start in warm_starts:
                    i += 1 
                    print("p" + str(i),hiden_layer_size, activation, solver, learning_rate, warm_start)
                    