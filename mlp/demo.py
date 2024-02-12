
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time 

start_time = time.time()
data_sizes = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
data_base_path = "/home/rouf/Documents/unsw_data/data"
devices = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 18, 19, 20, 22, 23, 26, 27, 28, 31]
train_test_split_fraction = 0.2

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
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = np.array(Y)
    return X, Y


X, y = load_data(1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Define the parameter grid for the MLPClassifier
#  (10, 100), (50, 100), (10, 50, 100)
#  0.01
# ''

param_grid = {
    'hidden_layer_sizes': [(50), (100), (200), (50, 100), (100, 200), (50, 100, 200)], 
    'activation': ['identity','relu', 'tanh'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.1, 0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'adaptive'],
}

# param_grid = {
#     'hidden_layer_sizes': [ (100)], 
#     'activation': ['identity','relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.01 ],
#     'learning_rate': ['constant', 'adaptive'],
# }

# Create MLP classifier
mlp = MLPClassifier(max_iter=100)

# Instantiate GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, cv=5)

# Fit the GridSearchCV instance to the data
grid_search.fit(X_train, y_train)

# # Print the best parameters found
# print("Best parameters found:")
# print(grid_search.best_params_)

# # Evaluate the best model on the test set
# best_model = grid_search.best_estimator_

# test_accuracy = best_model.score(X_test, y_test)
# print("Test accuracy of the best model:", test_accuracy)


# Get all mean test scores and parameter combinations
mean_test_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

# Find the index of the parameter combination with the best accuracy
best_index = np.argmax(mean_test_scores)
# Extract the best accuracy and corresponding parameters
best_accuracy = mean_test_scores[best_index]
best_parameters = params[best_index]

# Find the index of the parameter combination with the worst accuracy
worst_index = np.argmin(mean_test_scores)
# Extract the worst accuracy and corresponding parameters
worst_accuracy = mean_test_scores[worst_index]
worst_parameters = params[worst_index]

# Print the best accuracy and corresponding parameters
print("Best accuracy found:", best_accuracy)
print("Parameters corresponding to the best accuracy:", best_parameters)

# Print the worst accuracy and corresponding parameters
print("Worst accuracy found:", worst_accuracy)
print("Parameters corresponding to the worst accuracy:", worst_parameters)

end_time = time.time()
elapsed_time = end_time - start_time
print("total time taken:", (elapsed_time / 60), " minutes")