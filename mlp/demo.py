
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import time 
from collections import Counter

split_fraction = 0.2
random_number = 42
start_time = time.time()
data_sizes = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
devices = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 18, 19, 20, 22, 23, 26, 27, 28, 31]
data_base_path = "/home/rouf-linux/unsw_data/data"
maximum_iteration = 100

def load_data(data_size):
    data_path = f"{data_base_path}/{data_size}"
    X = []
    Y = []
    for device in devices:
        for sample in range(data_size):
            df = pd.read_csv(f"{data_path}/{device}/{sample}.csv")
            X.append(df.values.flatten().tolist())
            Y.append(device)
    X = np.array(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = np.array(Y)
    return X, Y


def split_data(X, y):
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=split_fraction, random_state=random_number)
    train_indices, test_indices = next(stratified_splitter.split(X, y))
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

X, y = load_data(200)
X_train, X_test, y_train, y_test = split_data(X, y)

# class_counts = Counter(y_test)

# param_grid = {
#     'hidden_layer_sizes': [(50, 100, 150)], 
#     'activation': ['relu'],
#     'solver': ['lbfgs'],
#     'alpha': [0.01],
#     'learning_rate': ['invscaling'],
#     'warm_start': [True]
# }

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (50, 100), (50, 150), (100, 50), (100, 100), (100, 150), (150, 50), (150, 100), (150, 150), (50, 50, 50), (50, 50, 100), (50, 50, 150), (50, 100, 50), (50, 100, 100), (50, 100, 150), (50, 150, 50), (50, 150, 100), (50, 150, 150), (100, 50, 50), (100, 50, 100), (100, 50, 150), (100, 100, 50), (100, 100, 100), (100, 100, 150), (100, 150, 50), (100, 150, 100), (100, 150, 150), (150, 50, 50), (150, 50, 100), (150, 50, 150), (150, 100, 50), (150, 100, 100), (150, 100, 150), (150, 150, 50), (150, 150, 100), (150, 150, 150)], 
    'activation': ['identity','relu', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'warm_start': [True, False]
}




mlp = MLPClassifier(max_iter=maximum_iteration)
grid_search = GridSearchCV(mlp, param_grid, cv=5)
grid_search.fit(X_train, y_train)


print("Best parameters found:")
print(grid_search.best_params_)
# Get all mean test scores and parameter combinations
mean_test_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
best_index = np.argmax(mean_test_scores)
best_accuracy = mean_test_scores[best_index]
best_parameters = params[best_index]
worst_index = np.argmin(mean_test_scores)
worst_accuracy = mean_test_scores[worst_index]
worst_parameters = params[worst_index]
print("Best accuracy found:", best_accuracy)
print("Parameters corresponding to the best accuracy:", best_parameters)
print("Worst accuracy found:", worst_accuracy)
print("Parameters corresponding to the worst accuracy:", worst_parameters)

end_time = time.time()
elapsed_time = end_time - start_time
print("total time taken:", (elapsed_time / 60), " minutes")