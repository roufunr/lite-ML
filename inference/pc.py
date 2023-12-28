import json
import os
import csv
import pandas
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import time
import logging
import sys
import requests
import zipfile
import shutil

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')


logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= root_path + '/logger/infer_'+ str(round(time.time())) +'.log',
                    filemode='a')

logger = logging.getLogger(__name__)


dataset_path = home_path + '/TensorflowLiteData/F/'

def mean(data):
    return sum(data) / len(data)

def median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

def percentile(data, percentile):
    size = len(data)
    sorted_data = sorted(data)
    index = (size * percentile) // 100
    return sorted_data[int(index)]

def variance(data):
    data_mean = mean(data)
    return sum((x - data_mean) ** 2 for x in data) / len(data)

def skewness(data):
    data_mean = mean(data)
    data_variance = variance(data)
    n = len(data)
    return (sum((x - data_mean) ** 3 for x in data) / n) / (data_variance ** 1.5)

def kurtosis(data):
    data_mean = mean(data)
    data_variance = variance(data)
    n = len(data)
    return (sum((x - data_mean) ** 4 for x in data) / n) / (data_variance ** 2) - 3

def load_json_to_dict(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def load_data():
    device_mapper = load_json_to_dict(home_path + "/lite-ML/device_mapper.json")
    test_sample_idx = [x for x in range(1, 20 + 1)]     # for testing accuracy change range from 16 to 20 + 1
    X = []
    Y = []
    for device_id in device_mapper:
        for sample_idx in test_sample_idx:
            csv_file_name = str(device_id) + '_' + str(sample_idx) + '.csv'
            df = pandas.read_csv(dataset_path + csv_file_name)
            two_d_list = df.values.tolist()
            one_d_list = [item for sublist in two_d_list for item in sublist]
            X.append(one_d_list)
            y = [0 for i in range(len(device_mapper))]
            y[device_mapper[device_id]] = 1
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    logger.info("Loaded data shape X:" + str(X.shape) + " Y:" + str(Y.shape))
    return X, Y

def download_model(model_idx):
    SERVER_IP = '192.168.1.188'
    PORT = 8000
    url = f"http://{SERVER_IP}:{PORT}?model_idx={model_idx}"
    response = requests.get(url)
    if response.status_code == 200:
        save_dir = f"{home_path}/lite-ML/downloaded_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"model_{model_idx}.zip")
        with open(save_path, 'wb') as file:
            file.write(response.content)
        logger.info(f"Model {model_idx} downloaded and saved to {save_path}")
        logger.info(f"Size of downloaded file: {os.path.getsize(save_path)} bytes")
        extract_path = f"{home_path}/lite-ML/downloaded_models/model_{model_idx}"
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"Model {model_idx} unzipped to {extract_path}")
    else:
        logger.info(f"Failed to download model {model_idx}. Server response: {response.status_code} - {response.reason}")

def delete_model(model_idx):
    os.remove(f"{home_path}/lite-ML/downloaded_models/model_{model_idx}.zip")
    shutil.rmtree(f"{home_path}/lite-ML/downloaded_models/model_{model_idx}")


def measure_inference_time_on_tf_model(X, Y, tf_model):
    total_data_points = len(Y)
    elapsed_time = []
    for i in range(total_data_points):
        start_time = time.time() * 1000
        tf_model.predict(np.array([X[i]]))
        end_time = time.time() * 1000
        elapsed_time.append(end_time - start_time)
    
    inference_times = elapsed_time
    metrics = {
        "average": mean(inference_times),
        "median": median(inference_times),
        "min": min(inference_times),
        "max": max(inference_times),
        "25th percentile": percentile(inference_times, 25),
        "75th percentile": percentile(inference_times, 75),
        "variance": variance(inference_times),
        "range": max(inference_times) - min(inference_times),
        "IQR": percentile(inference_times, 75) - percentile(inference_times, 25),
        "skewness": skewness(inference_times),
        "kurtosis": kurtosis(inference_times)
    }
    return metrics
        
def measure_inference_time_on_lite_model(X, Y, interpreter):
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]['index']
    total_data_points = len(Y)
    elapsed_time = []
    for i in range(total_data_points):
        interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))
        start_time = time.time() * 1000
        interpreter.invoke()
        end_time = time.time() * 1000
        elapsed_time.append(end_time - start_time)
    
    inference_times = elapsed_time
    metrics = {
        "average": mean(inference_times),
        "median": median(inference_times),
        "min": min(inference_times),
        "max": max(inference_times),
        "25th percentile": percentile(inference_times, 25),
        "75th percentile": percentile(inference_times, 75),
        "variance": variance(inference_times),
        "range": max(inference_times) - min(inference_times),
        "IQR": percentile(inference_times, 75) - percentile(inference_times, 25),
        "skewness": skewness(inference_times),
        "kurtosis": kurtosis(inference_times)
    }
    return metrics
    
X, Y = load_data()
model_idx = sys.argv[1]
download_model(model_idx)
tf_model = tf.keras.models.load_model(home_path + "/lite-ML/downloaded_models/model_"+ str(model_idx) +"/tf")
lite_model = tf.lite.Interpreter(model_path= home_path + "/lite-ML/downloaded_models/model_"+ str(model_idx) +"/" +str(model_idx)+ ".tflite")
measure_inference_time_on_tf_model(X, Y, tf_model)
measure_inference_time_on_lite_model(X, Y, lite_model)
delete_model(model_idx)


# def run_inference_on_tf_model(X, Y, tf_model):
#     predictions = np.array([])
#     total_data_points = len(Y)
#     for i in range(total_data_points):
#         prediction = tf_model.predict(np.array([X[i]]))
#         if predictions.size == 0:
#             predictions = np.array([prediction[0]])
#         else:
#             predictions = np.vstack((predictions, prediction[0]))

#     one_hot_predictions = []
#     for prediction in predictions:
#         one_hot_prediction = np.zeros_like(prediction)
#         one_hot_prediction[np.argmax(prediction)] = 1
#         one_hot_predictions.append(one_hot_prediction.tolist())
#     one_hot_predictions = np.array(one_hot_predictions)

#     predicted_labels = one_hot_predictions

#     # Accuracy
#     accuracy = metrics.accuracy_score(Y, predicted_labels)
#     print(f'Accuracy: {accuracy}')

#     # Precision, Recall, F1-score (for each class in a multi-class setting)
#     precision = metrics.precision_score(Y, predicted_labels, average='weighted')  # Change average to 'micro', 'macro', or 'weighted' for multi-class
#     recall = metrics.recall_score(Y, predicted_labels, average='weighted')
#     f1 = metrics.f1_score(Y, predicted_labels, average='weighted')

#     return precision, recall, f1

# def run_inference_on_lite_model(X, Y, interpreter):
#     interpreter.allocate_tensors()
#     input_tensor_index = interpreter.get_input_details()[0]['index']
#     predictions = np.array([])
#     for i in range(len(Y)):
#         interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))  # Ensure input data is FLOAT32
#         interpreter.invoke()
#         output_tensor_index = interpreter.get_output_details()[0]['index']
#         prediction = interpreter.get_tensor(output_tensor_index)
#         if predictions.size == 0:
#             predictions = np.array([prediction[0]])
#         else:
#             predictions = np.vstack((predictions, prediction[0]))
    
#     one_hot_predictions = []
#     for prediction in predictions:
#         one_hot_prediction = np.zeros_like(prediction)
#         one_hot_prediction[np.argmax(prediction)] = 1
#         one_hot_predictions.append(one_hot_prediction.tolist())
#     one_hot_predictions = np.array(one_hot_predictions)

#     predicted_labels = one_hot_predictions


#     accuracy = metrics.accuracy_score(Y, predicted_labels)
#     print(f'Accuracy: {accuracy}')

#     precision = metrics.precision_score(Y, predicted_labels, average='micro')  # Change average to 'micro', 'macro', or 'weighted' for multi-class
#     recall = metrics.recall_score(Y, predicted_labels, average='micro')
#     f1 = metrics.f1_score(Y, predicted_labels, average='micro')

#     return precision, recall, f1

# s_time = time.time()
# X, Y = load_test_data()
# csv_report = []
# start_model = int(sys.argv[1])
# end_model = int(sys.argv[2])
# for i in range(start_model, end_model + 1):
#     precision_tf, recall_tf, f1_tf = run_prediction(X,Y, i)
#     precision_lite, recall_lite, f1_lite = run_inference_lite(X,Y, i)
#     csv_report.append([i, precision_tf, precision_lite, recall_tf, recall_lite, f1_tf, f1_lite])
#     logger.info("DONE ::: " + str(i))

# file_path = 'metric.csv'

# # Writing the 2D list to a CSV file
# with open(file_path, 'a', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerows(csv_report)

# print(f"Data has been written to {file_path}")
# e_time = time.time()
# logger.info(str(start_model) + " to " + str(end_model) + " takes " + str(e_time - s_time) + " seconds")

    

