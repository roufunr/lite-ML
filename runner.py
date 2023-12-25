# runner.py

import subprocess
import time

def run_model_generation(param1, param2):
    command = ["python", "train_models.py", str(param1), str(param2)]
    subprocess.run(command)

if __name__ == "__main__":
    # Set the parameters
    start_idx = 11400
    offset = 25

    # Run the command in a while loop with a sleep of 5 seconds
    while True:
        if start_idx < 13825:
            run_model_generation(start_idx, offset)
            time.sleep(2)
            start_idx += offset
        else: 
            break
