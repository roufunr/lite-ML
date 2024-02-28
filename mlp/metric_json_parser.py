import json
import pandas as pd

# Assuming you have a JSON file named data.json
with open('/home/rouf-linux/lite-ML/mlp/PI_precision_recall_f1_time.json', 'r') as file:
    data_dict = json.load(file)

results_list = data_dict["results"]
total_len = len(results_list)
rows = []
header = [
    'idx', 
    'no_of_layers', 
    'l1', 
    'l2', 
    'l3', 
    'activation', 
    'solver', 
    'learning_rate', 
    'alpha', 
    'warm_start', 
    'precision', 
    'recall', 
    'f1', 
    'training time (ms)', 
    'inference time (ms)'
]

rows.append(header)
for i in range(total_len):
    number_of_hidden_layers = len(results_list[i]['hidden_layer_sizes'])
    hidden_layer_sizes = results_list[i]['hidden_layer_sizes']
    hidden_layer_sizes += [0 for i in range(3 - number_of_hidden_layers)] if number_of_hidden_layers < 3 else []

    row = [
        i + 1,
        number_of_hidden_layers,
        hidden_layer_sizes[0],
        hidden_layer_sizes[1],
        hidden_layer_sizes[2],
        results_list[i]['activation'],
        results_list[i]['solver'],
        results_list[i]['learning_rate'],
        results_list[i]['alpha'],
        results_list[i]['warm_start'],
        results_list[i]['precision'],
        results_list[i]['recall'],
        results_list[i]['f1'],
        results_list[i]['train_time'],
        results_list[i]['inference_time']
    ]
    rows.append(row)

print(rows)

df = pd.DataFrame(rows)
df.to_csv("metric_pi.csv")
