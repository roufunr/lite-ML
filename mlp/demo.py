import os
import shutil
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load data
X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Define the parameter grid for the MLPClassifier
#  (10, 100), (50, 100), (10, 50, 100)
#  0.01
# 'logistic'
param_grid = {
    'hidden_layer_sizes': [(100, 100, 100)], 
    'activation': ['relu', 'tanh'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.1 ],
    'learning_rate': ['constant', 'adaptive'],
}

# Create MLP classifier
mlp = MLPClassifier(max_iter=50)

# Instantiate GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, cv=5)

# Fit the GridSearchCV instance to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found:")
print(grid_search.best_params_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test accuracy of the best model:", test_accuracy)
