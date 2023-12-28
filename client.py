import requests
import os
import zipfile
import shutil
import time





def download_and_unzip_model(model_idx):
    SERVER_IP = '192.168.1.188'
    PORT = 8000
    url = f"http://{SERVER_IP}:{PORT}?model_idx={model_idx}"

    # Make a GET request to the server
    response = requests.get(url)

    if response.status_code == 200:
        # Create a directory to save the downloaded model
        save_dir = f"downloaded_models"
        os.makedirs(save_dir, exist_ok=True)

        # Save the received content (zip file) to the local directory
        save_path = os.path.join(save_dir, f"model_{model_idx}.zip")
        with open(save_path, 'wb') as file:
            file.write(response.content)

        print(f"Model {model_idx} downloaded and saved to {save_path}")

        # Print size of the downloaded file for debugging
        print(f"Size of downloaded file: {os.path.getsize(save_path)} bytes")

        extract_path = f"downloaded_models/model_{model_idx}"
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            # Extract the contents of the zip file to a temporary directory
            zip_ref.extractall(extract_path)

        print(f"Model {model_idx} unzipped to {extract_path}")
        
    else:
        print(f"Failed to download model {model_idx}. Server response: {response.status_code} - {response.reason}")


model_index = 7
download_and_unzip_model(model_index)
