import os
import time
import sys
import random
import csv
import cProfile


def custom_train_test_split(X, y, test_size=0.25, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    combined = list(zip(X, y))
    random.shuffle(combined)
    split_idx = int(len(X) * (1 - test_size))
    X_train, y_train = zip(*combined[:split_idx])
    X_test, y_test = zip(*combined[split_idx:])
    return list(X_train), list(X_test), list(y_train), list(y_test)

class NilsimsaClassifier:
    def __init__(self):
        self.training_data = {}

    def hex_to_zeros_similarity(self, hex_strings):
        binary_strings = [bin(int(h, 16))[2:].zfill(256) for h in hex_strings]
        zeros_similarity = [0] * 256
        for binary in binary_strings:
            for i, bit in enumerate(binary):
                if bit == '0':
                    zeros_similarity[i] += 1

        return zeros_similarity
    
    def train(self, hashes, devices):
        for hash, device in zip(hashes, devices):
            digest = hash
            if device not in self.training_data:
                self.training_data[device] = []
            self.training_data[device].append(digest)
        
        for(device, digests) in self.training_data.items():
            zero_similarity = self.hex_to_zeros_similarity(digests)
            self.training_data[device] = zero_similarity


    # @profile
    def predict(self, test_data_point):
        results = []
        for test_point in test_data_point:
           
            max_label = "none"
            max_score = -sys.maxsize

            test_digest = bin(int(test_point, 16))[2:].zfill(256)

            for label, training_zero_similarity in self.training_data.items():
                cum_score = 0
                for i in range(256):
                    if test_digest[i] == '0':
                        cum_score += training_zero_similarity[i]
                    else:
                        cum_score += (len(training_zero_similarity) - training_zero_similarity[i])
                
                if cum_score > max_score:
                    max_score = cum_score
                    max_label = label

            results.append(max_label)

        return results


def load_and_split_data(parent_folder, pid):
    data = []

    for subdir in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subdir)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith(".csv"):
                    device_label = int(file.split("-")[0])
                    file_path = os.path.join(subfolder_path, file)

                    with open(file_path, newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            if int(row[0]) == pid:
                                data.append((row[1], device_label))

    # Splitting data for each device
    X_train, X_test, y_train, y_test = [], [], [], []
    device_labels = set(label for _, label in data)

    for device_label in device_labels:
        device_data = [(x, y) for x, y in data if y == device_label]
        X_device, y_device = zip(*device_data)

        if len(X_device) >= 20:
            X_train_device, X_test_device, y_train_device, y_test_device = custom_train_test_split(
                X_device, y_device, test_size=0.25, random_state=42)
            
            X_train.extend(X_train_device)
            X_test.extend(X_test_device)
            y_train.extend(y_train_device)
            y_test.extend(y_test_device)

    return X_train, X_test, y_train, y_test

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

def measure_inference_time(classifier, X_test):
    inference_times = []

    for test_point in X_test:
        start_time = time.time()
        classifier.predict([test_point])
        end_time = time.time()
        inference_times.append(end_time - start_time)

    # Calculating various statistical metrics
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

# @profile


parent_folder = "Hashes"  # Replace with the path to your parent folder
print("Loading and splitting data...")
X_train, X_test, y_train, y_test = load_and_split_data(parent_folder, pid = 630)
print("Done with data loading and splitting")
# print(X_train)


def write_metrics_to_csv(metrics, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])  # Write the header

        for key, value in metrics.items():
            writer.writerow([key, value])

classifier = NilsimsaClassifier()
classifier.train(X_train, y_train)



def generate_inference_time():
    inferece_metrics = measure_inference_time(classifier, X_test)
    write_metrics_to_csv(inferece_metrics, 'inference_metrics.csv')
    print("Inference Metrics written to 'inference_metrics.csv'")


# main_predict(classifier, X_test)
def run_predict():
    classifier.predict(X_test)

cProfile.run('run_predict()')

# generate_inference_time()
