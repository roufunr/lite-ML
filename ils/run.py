import numpy as np
import time
import psutil
import datetime
import statistics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from multiprocessing import Process, Lock, Manager,Queue, Pipe
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os import listdir
from os.path import isfile, join
import copy
X_train = []
Y_train = []
X_test = []
Y_test = []
feature_names = []
feature_values = []

weights = {
    'accuracy': 0.5,  # Example weight, adjust based on the importance of accuracy
    'time': 0.25,  # Adjust based on the importance of minimizing time
    'memory': 0.25,  # Adjust based on the importance of minimizing memory usage
}

def dataReader(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for files in onlyfiles:
        label=files.split("_")[0]
        measurementID=int(files.split("_")[1].split(".csv")[0])
        #print(files)
        #print(label, measurementID)

        featuresString=""
        r=open(path+files)
        for line in r:
            line=line.strip("\n\r")
            if ("arp" in line):
                continue
            featuresString=featuresString+line+","

        featuresString=featuresString.strip(",")
        splitted=featuresString.split(",")
        splittedInt=list(map(int, splitted))

        if(measurementID>15): #goes to test
            X_test.append(splittedInt)
            Y_test.append(label)
        else:                   #goes to train
            X_train.append(splittedInt)
            Y_train.append(label)
           # print(Y_train)


def inference_time_predicted_values(clf):
    inferenceTime_s = []
    predictedValues = []

    for i in range(0, len(X_test)):  # for every instance

        start_time = datetime.datetime.now()
        prediction = clf.predict([X_test[i]])
        delta = datetime.datetime.now() - start_time
        inferenceTime = delta.total_seconds() * 1000
        inferenceTime_s.append(inferenceTime)
#        acc = accuracy_score(Y_test, prediction)
        prediction_list = prediction.tolist()
        predictedValues = predictedValues + prediction_list

    accuracy = precision_recall_fscore_support(Y_test, predictedValues, average='macro')
    precision = accuracy[0]
    recall = accuracy[1]
    f1score = accuracy[2]

    median_time = statistics.median(inferenceTime_s)
    # print(median_time,precision,recall,f1score)
    return [median_time, precision, recall, f1score]


def infer(clf, td, iteration_number):
    clf.n_jobs=1
    i = 0
    while i < iteration_number:
        clf.predict([td])
        i+=1


def cpu_memory_monitor(clf, iteration_number, sleep_time):
    num_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    cpu_percents_s = []
    mem_usage_s = []

    for i in range(0, len(X_test)):  # for every instance
        worker_process = Process(target=infer, args=(clf, X_test[i], iteration_number))
        worker_process.start()
        p = psutil.Process(worker_process.pid)

        while worker_process.is_alive():
            try:
                memory_info = p.memory_info()
                cpu_info = p.cpu_percent()

                memory_usage_mb = memory_info.rss / (1024 ** 2)
                normalized_cpu_percent = cpu_info / float(num_cores)

                cpu_percents_s.append(normalized_cpu_percent)
                mem_usage_s.append(memory_usage_mb)
            except:
                aa = 2

            if (sleep_time != 0):
                time.sleep(sleep_time)

        worker_process.join()

        if (i == 0):
            print(len(cpu_percents_s), len(mem_usage_s))
            # print (cpu_percents_s)

    returnMemory = -1
    returnCPU = -1
    if (len(cpu_percents_s) != 0):
        returnCPU = statistics.median(cpu_percents_s)
    if (len(mem_usage_s) != 0):
        returnMemory = statistics.median(mem_usage_s)

    return [returnCPU, returnMemory]


def getModelMLP(current_feature):
    # hidden_layer_sizes = current_feature[1]
    # activation = current_feature[2]
    # solver = current_feature[3]
    # learning_rate = current_feature[4]
    warm_start = current_feature['warm_start']
    #
    # hidden_layer_sizes = int(hidden_layer_sizes)
    # warm_start = eval(str(warm_start))
    sizes = current_feature['hidden_layer_sizes']
    rate = current_feature['learning_rate']
    #print(rate)
    clf = MLPClassifier(hidden_layer_sizes=sizes, learning_rate=rate, warm_start=warm_start)
    #print(clf)
    clf.fit(X_train, Y_train)
    return clf


def evaluate_model(params):

    model = getModelMLP(params)
    #print(model)
    #print(model.get_params())
    time_accuracy = inference_time_predicted_values(model)
    cpu_memory = cpu_memory_monitor(model, iteration_number, sleep_time)
    #return time_accuracy[0], time_accuracy[4], cpu_memory[1]
    score = (weights['accuracy'] * time_accuracy[2]) - (weights['time'] * time_accuracy[0]) - (weights['memory'] * cpu_memory[1])
    print("score", score)
    return score, time_accuracy[2], time_accuracy[0], cpu_memory[1]

# def perturb_solution(current_solution):
#
#     new_solution = copy.deepcopy(current_solution)
#     print("perturb")
#     if 'hidden_layer_sizes' in new_solution:
#         new_hidden_layer_sizes = []
#         for size in new_solution['hidden_layer_sizes']:
#             # Adjust each layer size by up to ±10 neurons, but ensure at least 1 neuron remains
#             new_size = max(1, size + np.random.randint(-5, 6))
#             new_hidden_layer_sizes.append(new_size)
#
#         # Randomly add or remove a layer with certain probability
#         if np.random.rand() > 0.5:  # 50% chance to modify the number of layers
#             if np.random.rand() > 0.5 and len(new_hidden_layer_sizes) > 1:
#                 # Remove a layer, ensuring at least one layer remains
#                 del new_hidden_layer_sizes[np.random.randint(0, len(new_hidden_layer_sizes))]
#             else:
#                 # Add a new layer with a size between 1 and 50 neurons as an example
#                 new_hidden_layer_sizes.append(np.random.randint(1, 51))
#
#         new_solution['hidden_layer_sizes'] = tuple(new_hidden_layer_sizes)
#
#     # if 'learning_rate_init' in new_solution:
#     #     # Scale the learning rate by a factor between 0.5 to 1.5, for example
#     #     new_learning_rate = new_solution['learning_rate_init'] * (0.5 + np.random.rand())
#     #     # Ensure the learning rate stays within the predefined range
#     #     new_learning_rate = min(max(new_learning_rate, 0.001), 0.1)
#     #     new_solution['learning_rate_init'] = new_learning_rate
#     learning_rate_schedules = ['constant', 'invscaling', 'adaptive']
#     new_solution['learning_rate'] = np.random.choice(learning_rate_schedules)
#
#     return new_solution

def perturb_solution(current_solution, perturbation_rules):
    new_solution = copy.deepcopy(current_solution)

    # Loop over each parameter in the solution
    for parameter, rules in perturbation_rules.items():
        if parameter in new_solution:
            if rules['type'] == 'numeric':
                new_solution[parameter] += np.random.uniform(-rules['range'], rules['range'])
                new_solution[parameter] = np.clip(new_solution[parameter], rules['min'], rules['max'])
            elif rules['type'] == 'categorical':
                new_solution[parameter] = np.random.choice(rules['options'])
            elif rules['type'] == 'integer':
                new_solution[parameter] += np.random.randint(-rules['range'], rules['range'] + 1)
                new_solution[parameter] = int(np.clip(new_solution[parameter], rules['min'], rules['max']))
            elif rules['type'] == 'boolean':
                new_solution[parameter] = np.random.choice(rules['options'])

    return new_solution
def scalar_objective_function(accuracy, time, memory, weights):
    return (weights['accuracy'] * accuracy) - (weights['time'] * time) - (weights['memory'] * memory)


def local_search(best_solution):
    #time, accuracy, memory = evaluate_model(base_solution, X_train, y_train, X_val, y_val)
   # best_score = evaluate_model(best_solution)
    # Perturb the solution to generate a new candidate solution
    #new_solution = perturb_solution(base_solution)
    print("local")
    # Evaluate the new solution
    best_score,_,_,_ = evaluate_model(best_solution)
    for _ in range(5):
        #new_solution = perturb_solution(best_solution)
        new_solution = perturb_solution(best_solution, perturbation_rules)
        #print("perturb:", new_solution)
        new_score,_,_,_ = evaluate_model(new_solution)

        if new_score > best_score:
            best_solution, best_score = new_solution, new_score

    return best_solution, best_score


def iterated_local_search(max_iterations, best_solution):
    # Initialize with a base solution

    best_score,_,_,_ = evaluate_model(best_solution)

    for iteration in range(max_iterations):
        #new_solution = perturb_solution(best_solution)
        new_solution = perturb_solution(best_solution, perturbation_rules)
        print(new_solution)
        new_solution, new_score = local_search(new_solution)
        print(iteration)
        # Evaluate new solution and update best_solution if it is better
        if new_score >= best_score:
            best_solution = new_solution
            best_score = new_score
    score, f1, time, mem = evaluate_model(best_solution)
    print("Best solution:" , best_solution)
    print("Best score:", best_score)
    print("F1, Time, Memory:", f1, time, mem)


if __name__ == '__main__':

    sleep_time=0.005    #decreasing this will increase frequency of our measurement
    iteration_number=500   #increasing it will increase overall execution time, but we will be able to make more measurements

    #path="./parameters/MLP_parameters.csv"

    dataReader("./F/")
    base_solution = {'hidden_layer_sizes': 80, 'learning_rate': 'adaptive', 'warm_start' : True}
    perturbation_rules = {
        'hidden_layer_sizes': {
            'type': 'integer',
            'range': 20,  # how much the parameter can change by
            'min': 20,  # minimum allowed value
            'max': 180  # maximum allowed value
        },
        'learning_rate': {
            'type': 'categorical',
            'options': ['constant', 'invscaling', 'adaptive']
        },
        'warm_start': {
            'type': 'boolean',
            'options': [True, False]
        }
    }

    iterated_local_search(10, base_solution)
