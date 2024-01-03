import re
import logging
import os
import pandas as pd
import csv 

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/analysis.log',
                    filemode='a')

logger = logging.getLogger(__name__)

def extract_numbers(line):
    """Extract numerical values from a given line."""
    return [float(num) for num in re.findall(r'\d+\.\d+|\d+', line)]

def find_line_with_key(file_lines, key_phrase):
    lines = []
    for line in file_lines:
        if key_phrase in line:
            lines.append(line)
    return lines

def extract_profiling_data(file_path):
    """Extract profiling data from the specified file."""
    with open(file_path, 'r') as file:
        file_content = file.readlines()

    # Key phrases to search in the file
    tf_model_cpu_time_key = "tf_model.predict(np.array([X[i]])"
    lite_model_set_cpu_time_key = "interpreter.set_tensor(input_tensor_index, np.array([X[i]]).astype(np.float32))"
    lite_model_invoke_cpu_time_key = "interpreter.invoke()"
    tf_model_cumtime_key = "profiler.py:56(measure_on_tf_model)"
    lite_model_cumtime_key = "profiler.py:61(measure_on_lite_model)"

    # Extracting required lines
    line_tf_model_cpu_time = find_line_with_key(file_content, tf_model_cpu_time_key) #float(re.findall(r'\S+', line_tf_model_cpu_time[0])[2])
    line_lite_model_set_cpu_time = find_line_with_key(file_content, lite_model_set_cpu_time_key) #float(re.findall(r'\S+', line_lite_model_set_cpu_time[0])[2])
    line_lite_model_invoke_cpu_time = find_line_with_key(file_content, lite_model_invoke_cpu_time_key) #float(re.findall(r'\S+', line_lite_model_invoke_cpu_time[0])[2])
    line_tf_model_cumtime = find_line_with_key(file_content, tf_model_cumtime_key) #float(re.findall(r'\S+', line_tf_model_cumtime[0])[3])
    line_lite_model_cumtime = find_line_with_key(file_content, lite_model_cumtime_key) #float(re.findall(r'\S+', line_lite_model_cumtime[0])[3])

    # Extracting and reporting the values
    profile_data = {
        "tf": {
            "line": float(re.findall(r'\S+', line_tf_model_cpu_time[0])[2]),
            "cum": float(re.findall(r'\S+', line_tf_model_cumtime[0])[3]),
            "mem": float(re.findall(r'\S+', line_tf_model_cpu_time[1])[3])
        },
        "lite": {
            "line": float(re.findall(r'\S+', line_lite_model_set_cpu_time[0])[2]) + float(re.findall(r'\S+', line_lite_model_invoke_cpu_time[0])[2]),
            "cum": float(re.findall(r'\S+', line_lite_model_cumtime[0])[3]),
            "mem": float(re.findall(r'\S+', line_lite_model_set_cpu_time[1])[3]) + float(re.findall(r'\S+', line_lite_model_invoke_cpu_time[1])[3])
        }
    }
    return profile_data

def get_median_from_csv(file_path, metric_column='Metric', value_column='Value', median_metric='median'):
    try:
        data = pd.read_csv(file_path)
        median_value = data[data[metric_column] == median_metric][value_column].iloc[0]
        return median_value
    except Exception as e:
        logger.info(f"Error occurred: {e}")
        return None
def write_2d_list_to_csv(data_2d, file_path):
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_2d)
        logger.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logger.info(f"Error occurred while writing to the file: {e}")
        
rows = [['model_idx', 'tf_time', 'lite_time', 'tf_cpu_line', 'lite_cpu_line', 'tf_cum_cpu', 'lite_cum_cpu', 'tf_mem', 'lite_mem']]
for i in range(1, 13825):
    file_path = home_path + f"/resource_utilization/{i}/profiling.txt"
    profiling_data = extract_profiling_data(file_path)
    tf_inference_time = get_median_from_csv(home_path + f"/resource_utilization/{i}/tf_time.csv")
    lite_inference_time = get_median_from_csv(home_path + f"/resource_utilization/{i}/lite_time.csv")
    row = [
        i, 
        tf_inference_time, 
        lite_inference_time, 
        profiling_data['tf']['line'],
        profiling_data['lite']['line'],
        profiling_data['tf']['cum'],
        profiling_data['lite']['cum'],
        profiling_data['tf']['mem'],
        profiling_data['lite']['mem']
    ]
    rows.append(row)
    logger.info(f"{i} --> DONE")

write_2d_list_to_csv(rows, home_path.split("/")[2] + "_" + "data.csv")


    
    





