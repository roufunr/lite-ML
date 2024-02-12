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
    # 'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), ], 
    'activation': ['identity','relu', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.01, 0.001, 0.0001],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'warm_start': [True, False]
}


os.makedirs(f"{home_path}/mlp_profiler_result/", exist_ok=True)
idx = 1
for hidden_layer_sizes, activation, solver, alpha, learning_rate, warm_start in product(*param_grid.values()):
    start_time = time.time()
    hidden_layer_sizes =  "_".join(map(str, hidden_layer_sizes))
    command1 = f'python cpu_util.py  {hidden_layer_sizes} {activation} {solver} {alpha} {learning_rate} {warm_start} > {home_path}/mlp_profiler_result/{idx}_cpu.txt'
    command2 = f'python mem_util.py  {hidden_layer_sizes} {activation} {solver} {alpha} {learning_rate} {warm_start} > {home_path}/mlp_profiler_result/{idx}_mem.txt'
    commands = [command1, command2]
    for cmd in commands:
        try:
            logger.info(f"{cmd} --- STARTED")
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"{cmd} --- DONE")
        except subprocess.CalledProcessError as e:
            logger.info(f"Command '{cmd}' failed with error: {e}")
    idx += 1
    end_time = time.time()
    
    logger.info(f"{idx} takes {end_time - start_time} s")