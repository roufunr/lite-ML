import subprocess
import os
import time
import logging

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/time_runner.log',
                    filemode='a')

logger = logging.getLogger(__name__)



 
for i in range(1, 13824, 128):
    start_time = time.time()
    # command1 = f'python3 inference_time.py {i}'
    command5 = f'python inference_time.py {i} {i + 127}'
    commands = [command5]
    
    for cmd in commands:
        try:
            print(f"{cmd} --- STARTED")
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"{cmd} --- DONE")
            print(f"{cmd} --- END")
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with error: {e}")
    
    if (i - 1) % 256 == 0: 
        logger.info(f"runner goes to sleep for 5 seconds")
        time.sleep(5)
    
    end_time = time.time()
    logger.info(f"from model-{i} to model-{i + 127} takes {end_time - start_time} seconds")