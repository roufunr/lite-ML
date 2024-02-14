import subprocess
import os
import time
import logging
from itertools import product

home_path = os.path.expanduser('~')

os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/mlp_profiler_runner.log',
                    filemode='a')

logger = logging.getLogger(__name__)
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (50, 100), (50, 150), (100, 50), (100, 100), (100, 150), (150, 50), (150, 100), (150, 150), (50, 50, 50), (50, 50, 100), (50, 50, 150), (50, 100, 50), (50, 100, 100), (50, 100, 150), (50, 150, 50), (50, 150, 100), (50, 150, 150), (100, 50, 50), (100, 50, 100), (100, 50, 150), (100, 100, 50), (100, 100, 100), (100, 100, 150), (100, 150, 50), (100, 150, 100), (100, 150, 150), (150, 50, 50), (150, 50, 100), (150, 50, 150), (150, 100, 50), (150, 100, 100), (150, 100, 150), (150, 150, 50), (150, 150, 100), (150, 150, 150)], 
    # 'hidden_layer_sizes': [(50, 50, 150)], 
    'activation': ['identity','relu', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'warm_start': [True, False]
}
params = []

os.makedirs(home_path + "/mlp_profiler_result", exist_ok=True)

idx = 1
for hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start in product(*param_grid.values()):
    
    hidden_layer_sizes =  "_".join(map(str, hidden_layer_sizes))
    params.append({
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'warm_start': warm_start
    })

total_len = len(params)
for idx in range(0, total_len):
    start_time = time.time()
    hidden_layer_sizes = params[idx]['hidden_layer_sizes']
    activation = params[idx]['activation']
    solver = params[idx]['solver']
    alpha = str(params[idx]['alpha'])
    learning_rate = params[idx]['learning_rate']
    warm_start = str(params[idx]['warm_start'])

    command1 = "python3 cpu_util.py " + " " + hidden_layer_sizes + " " + activation + " " + solver + " " + alpha + " " + learning_rate + " " + warm_start + " > " + home_path + "/mlp_profiler_result/" + str(idx + 1) + "_cpu.txt"
    command2 = "python3 mem_util.py " + " " + hidden_layer_sizes + " " + activation + " " + solver + " " + alpha + " " + learning_rate + " " + warm_start + " > " + home_path + "/mlp_profiler_result/" + str(idx + 1) + "_mem.txt"
    commands = [command1, command2]
    for cmd in commands:
        try:
            logger.info(cmd +  " -> STARTED")
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(cmd +  " -> ENDED")
        except subprocess.CalledProcessError as e:
            logger.info("Command " + cmd + " failed with error: " + str(e))
    end_time = time.time()
    logger.info(str(idx + 1) + " takes " + str(end_time - start_time) + " s")
    time.sleep(0.1)
