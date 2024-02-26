from line_profiler import LineProfiler
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import sys

split_fraction = 0.2
random_number = 42
data_sizes = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
devices = [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 18, 19, 20, 22, 23, 26, 27, 28, 31]
data_base_path = "/home/rouf/unsw_data/data"
maximum_iteration = 100

hidden_layer_sizes =  tuple([int(x) for x in sys.argv[1].split("_")])
activation = sys.argv[2]
solver = sys.argv[3]
alpha = float(sys.argv[4])
learning_rate = sys.argv[5]
warm_start = sys.argv[6].lower() == "true"

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

def get_data():
    X, y = load_data(200)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_data()

def run_mlp():
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                        activation=activation, 
                        solver=solver, 
                        alpha=alpha, 
                        learning_rate=learning_rate, 
                        warm_start=warm_start, 
                        max_iter=maximum_iteration)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    return y_pred
    
def profile_line(func, *args, **kwargs):
    profiler = LineProfiler()
    profiler.add_function(func)
    profiler.runcall(func, *args, **kwargs)
    profiler.print_stats()

profile_line(run_mlp)