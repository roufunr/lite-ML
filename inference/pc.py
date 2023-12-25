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


logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='/home/rouf-linux/lite-ML/infer.log',  # Specify the log file
                    filemode='a') 

logger = logging.getLogger(__name__)


dataset_path = '/home/rouf-linux/TensorflowLiteData/F/'
total_device = 23
def load_json_to_dict(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def load_test_data():
    device_mapper = load_json_to_dict("/home/rouf-linux/lite-ML/device_mapper.json")
    test_sample_idx = [x for x in range(16, 21)]
    X = []
    Y = []
    for device_id in device_mapper:
        for sample_idx in test_sample_idx:
            csv_file_name = str(device_id) + '_' + str(sample_idx) + '.csv'
            df = pandas.read_csv(dataset_path + csv_file_name)
            two_d_list = df.values.tolist()
            one_d_list = [item for sublist in two_d_list for item in sublist]
            X.append(one_d_list)
            y = [0 for i in range(total_device)]
            y[device_mapper[device_id]] = 1
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, Y

def run_prediction(X, Y, model_idx):
    tf_model = tf.keras.models.load_model("/home/rouf-linux/lite-ML/models/tf/" + str(model_idx))
    predictions = tf_model.predict(X)

    one_hot_predictions = []
    for prediction in predictions:
        one_hot_prediction = np.zeros_like(prediction)
        one_hot_prediction[np.argmax(prediction)] = 1
        one_hot_predictions.append(one_hot_prediction.tolist())
    one_hot_predictions = np.array(one_hot_predictions)

    predicted_labels = one_hot_predictions

    # Accuracy
    accuracy = metrics.accuracy_score(Y, predicted_labels)
    print(f'Accuracy: {accuracy}')

    # Precision, Recall, F1-score (for each class in a multi-class setting)
    precision = metrics.precision_score(Y, predicted_labels, average='weighted')  # Change average to 'micro', 'macro', or 'weighted' for multi-class
    recall = metrics.recall_score(Y, predicted_labels, average='weighted')
    f1 = metrics.f1_score(Y, predicted_labels, average='weighted')

    return precision, recall, f1

def run_inference_lite(X, Y, model_idx):
    interpreter = tf.lite.Interpreter(model_path="/home/rouf-linux/lite-ML/models/lite/" + str(model_idx) + ".tflite")
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]['index']
    predictions = np.array([])
    for i in range(len(Y)):
        interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))  # Ensure input data is FLOAT32
        interpreter.invoke()
        output_tensor_index = interpreter.get_output_details()[0]['index']
        prediction = interpreter.get_tensor(output_tensor_index)
        if predictions.size == 0:
            predictions = np.array([prediction[0]])
        else:
            # If 'a' is not empty, use numpy.vstack to stack the new row
            predictions = np.vstack((predictions, prediction[0]))
    
    print(predictions.shape)
    one_hot_predictions = []
    for prediction in predictions:
        one_hot_prediction = np.zeros_like(prediction)
        one_hot_prediction[np.argmax(prediction)] = 1
        one_hot_predictions.append(one_hot_prediction.tolist())
    one_hot_predictions = np.array(one_hot_predictions)

    predicted_labels = one_hot_predictions

    # Accuracy
    accuracy = metrics.accuracy_score(Y, predicted_labels)
    print(f'Accuracy: {accuracy}')

    # Precision, Recall, F1-score (for each class in a multi-class setting)
    precision = metrics.precision_score(Y, predicted_labels, average='micro')  # Change average to 'micro', 'macro', or 'weighted' for multi-class
    recall = metrics.recall_score(Y, predicted_labels, average='micro')
    f1 = metrics.f1_score(Y, predicted_labels, average='micro')

    return precision, recall, f1

s_time = time.time()
X, Y = load_test_data()
csv_report = []
start_model = int(sys.argv[1])
end_model = int(sys.argv[2])
for i in range(start_model, end_model + 1):
    precision_tf, recall_tf, f1_tf = run_prediction(X,Y, i)
    precision_lite, recall_lite, f1_lite = run_inference_lite(X,Y, i)
    csv_report.append([i, precision_tf, precision_lite, recall_tf, recall_lite, f1_tf, f1_lite])
    logger.info("DONE ::: " + str(i))

file_path = 'metric.csv'

# Writing the 2D list to a CSV file
with open(file_path, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(csv_report)

print(f"Data has been written to {file_path}")
e_time = time.time()
logger.info(str(start_model) + " to " + str(end_model) + " takes " + str(e_time - s_time) + " seconds")

    

