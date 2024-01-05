import subprocess
import os
import time
import logging

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/mem_prof.log',
                    filemode='a')

logger = logging.getLogger(__name__)
  
for i in range(1, 13824 + 1):
    start_time = time.time()
    os.makedirs(f"{home_path}/mem_utilization/", exist_ok=True)
    command4 = f'python lite_mem_profiler.py {i} > {home_path}/mem_utilization/{i}_lite.txt'
    command5 = f'python tf_mem_profiler.py {i} > {home_path}/mem_utilization/{i}_tf.txt'
    commands = [command4, command5]
    for cmd in commands:
        try:
            print(f"{cmd} --- STARTED")
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"{cmd} --- DONE")
            print(f"{cmd} --- END")
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with error: {e}")
    end_time = time.time()
    logger.info(f"model-{i} takes {end_time - start_time} seconds")
    if i % 256 == 0: 
        time.sleep(10)
    
    

