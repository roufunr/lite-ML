
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import time 
from itertools import product
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import json

# Configure logging
logging.basicConfig(filename='mlp.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


split_fraction = 0.2
random_number = 42
start_time = time.time()
data_sizes = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
devices = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 18, 19, 20, 22, 23, 26, 27, 28, 31]
data_base_path = "/home/rouf-linux/unsw/data"
maximum_iteration = 100

# param_grid = {
#     'hidden_layer_sizes': [(50, 100, 150)], 
#     'activation': ['relu'],
#     'solver': ['adam'],
#     'alpha': [0.01],
#     'learning_rate': ['adaptive'],
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

results = []
i = 1
for hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start in product(*param_grid.values()):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate, warm_start=warm_start, max_iter=maximum_iteration)
    
    train_time_start = time.time() * 1000
    mlp.fit(X_train, y_train)
    train_time_end = time.time() * 1000
    
    inference_time_start = time.time() * 1000
    y_pred = mlp.predict(X_test)
    inference_time_end = time.time() * 1000
    
    # Calculate precision, recall, and F1-score for each class
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    # Calculate overall precision, recall, and F1-score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    
    results.append({
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'warm_start': warm_start,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': tuple(precision_per_class),
        'recall_per_class': tuple(recall_per_class),
        'f1_per_class': tuple(f1_per_class),
        'train_time': train_time_end - train_time_start,
        'inference_time': inference_time_end - inference_time_start
    })

    logging.info(f"{i} th model done")
    logging.info(f"hidden_layer_sizes: {hidden_layer_sizes}, activation: {activation}, solver: {solver}, alpha: {alpha}, learning_rate: {learning_rate}, warm_starts: {warm_start}")
    logging.info(f"precision: {precision}, recall: {recall}, f1: {f1}")
    logging.info(f"training_time: {train_time_end - train_time_start}, inference_time: {inference_time_end - inference_time_start}")
    i+=1

final_result = {
    'results': results
}
with open('precision_recall_f1_time.json', 'w') as f:
    json.dump(final_result, f, indent=4)
    


