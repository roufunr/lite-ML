import subprocess
import os
import time
import requests
import zipfile
import shutil
import logging

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/runner.log',
                    filemode='a')

logger = logging.getLogger(__name__)


def download_model(model_idx):
    SERVER_IP = '192.168.1.188'
    PORT = 8000
    url = f"http://{SERVER_IP}:{PORT}?model_idx={model_idx}"
    response = requests.get(url)
    if response.status_code == 200:
        save_dir = f"{home_path}/downloaded_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"model_{model_idx}.zip")
        with open(save_path, 'wb') as file:
            file.write(response.content)
        logger.info(f"Model {model_idx} downloaded and saved to {save_path}")
        logger.info(f"Size of downloaded file: {os.path.getsize(save_path)} bytes")
        extract_path = f"{home_path}/downloaded_models/model_{model_idx}"
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"Model {model_idx} unzipped to {extract_path}")
    else:
        logger.info(f"Failed to download model {model_idx}. Server response: {response.status_code} - {response.reason}")

def delete_model(model_idx):
    os.remove(f"{home_path}/downloaded_models/model_{model_idx}.zip")
    shutil.rmtree(f"{home_path}/downloaded_models/model_{model_idx}")

def generate_result_dir(model_idx):
    os.makedirs(home_path + "/resource_utilization/" + str(model_idx), exist_ok = True)  
    os.makedirs(home_path + "/downloaded_models", exist_ok=True)
    logger.info("folder generation done!")

# command2 = f'kernprof -l -v line_and_memory_profile.py {i} > {home_path}/resource_utilization/{i}/line.txt'
# command3 = f'python3 -m memory_profiler line_and_memory_profile.py {i} > {home_path}/resource_utilization/{i}/memory.txt'
# command4 = f'python3 -m cProfile -s cumulative cumulative_profile.py {i} > {home_path}/resource_utilization/{i}/cpu.txt'
    
for i in range(1, 13824 + 1):
    
    start_time = time.time()
    # command1 = f'python3 inference_time.py {i}'
    command5 = f'python3 profiler.py {i} > {home_path}/resource_utilization/{i}/profiling.txt'
    commands = [command5]
    logging.info(f"Downloading {i} th model")
    generate_result_dir(i)
    download_model(i)
    logging.info(f"Downloading {i} th model ---  DONE")
    for cmd in commands:
        try:
            print(f"{cmd} --- STARTED")
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"{cmd} --- DONE")
            print(f"{cmd} --- END")
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with error: {e}")
    delete_model(i)
    if i % 16 == 0: 
        time.sleep(2)
    end_time = time.time()
    logger.info(f"model-{i} takes {end_time - start_time} seconds")

