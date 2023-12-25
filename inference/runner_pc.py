import subprocess
import time

def run_inference(param1, param2):
    command = ["python", "pc.py", str(param1), str(param2)]
    subprocess.run(command)
    
# Define the total number of models
total_models = 13824
batch_size = 256


for i in range(1, total_models + 1, batch_size):
    start_model = i
    end_model = min(i + batch_size - 1, total_models)

    run_inference(start_model, end_model)
    time.sleep(2)
