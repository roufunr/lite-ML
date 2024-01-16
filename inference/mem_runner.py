import subprocess
import os
import time
import logging
import shutil
import requests
import zipfile


home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/mem_prof.log',
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

for i in range(7319, 13824 + 1):
    start_time = time.time()
    download_model(i)
    os.makedirs(f"{home_path}/mem_utilization/", exist_ok=True)
    command4 = f'python3 lite_mem_profiler.py {i} > {home_path}/mem_utilization/{i}_lite.txt'
    command5 = f'python3 tf_mem_profiler.py {i} > {home_path}/mem_utilization/{i}_tf.txt'
    commands = [command4, command5]
    for cmd in commands:
        try:
            proc_start_time = time.time()
            subprocess.run(cmd, shell=True, check=True)
            proc_end_time = time.time()
            logger.info(f"{cmd} takes {proc_end_time - proc_start_time} seconds")
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with error: {e}")
    end_time = time.time()
    logger.info(f"model-{i} takes {end_time - start_time} seconds")
    
    if i % 128 == 0: 
        time.sleep(10)
    
    delete_model(i)