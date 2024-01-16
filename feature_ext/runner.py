import subprocess
import os
import time
import logging

home_path = os.path.expanduser('~')

os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/feature_profiler_runner.log',
                    filemode='a')

logger = logging.getLogger(__name__)
data_path = f'{home_path}/network_data'
files = os.listdir(data_path)


os.makedirs(f"{home_path}/feature_ext_profiler_result_new/", exist_ok=True)
idx = 1
for filename in files:
    start_time = time.time()
    pieces = filename.split("-")
    device_id = pieces[1]
    sample_id = pieces[2].split(".")[0]
    command1 = f'python opt_cpu_profiler.py {filename} > {home_path}/feature_ext_profiler_result_new/{device_id}_{sample_id}_cpu.txt'
    command2 = f'python opt_mem_profiler.py {filename} > {home_path}/feature_ext_profiler_result_new/{device_id}_{sample_id}_mem.txt'
    commands = [command1, command2]
    for cmd in commands:
        try:
            logger.info(f"{cmd} --- STARTED")
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"{cmd} --- DONE")
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with error: {e}")
    idx += 1
    end_time = time.time()
    
    logger.info(f"{idx} : {filename} takes {end_time - start_time} s")