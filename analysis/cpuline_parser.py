import logging
import os
import csv 

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/cpu_analysis.log',
                    filemode='a')

logger = logging.getLogger(__name__)

def parse_profile(i):
    profiler_file = f'{home_path}/resource_utilization/{i}/profiling.txt'
    # profiler_file = f'{home_path}/lite-ML/analysis/profiling.txt'
    with open(profiler_file, 'r') as file:
        profiler_lines = file.readlines()
    
    tf_cpu_line_text = profiler_lines[26].split(" ")
    tf_cpu_line_text = [x for x in tf_cpu_line_text if x != '']
    
    lite_allocate_tensor_text = profiler_lines[37].split(" ")
    lite_allocate_tensor_text = [x for x in lite_allocate_tensor_text if x != '']
    
    lite_get_input_tensor_text = profiler_lines[38].split(" ")
    lite_get_input_tensor_text = [x for x in lite_get_input_tensor_text if x != '']
    
    lite_set_input_data_text = profiler_lines[41].split(" ")
    lite_set_input_data_text = [x for x in lite_set_input_data_text if x != '']
    
    lite_invoke_text = profiler_lines[42].split(" ")
    lite_invoke_text = [x for x in lite_invoke_text if x != '']
    
    tf_cpu_lines = float(tf_cpu_line_text[2])/5
    
    lite_cpu_lines = float(lite_allocate_tensor_text[2]) 
    + float(lite_get_input_tensor_text[2]) 
    + (float(lite_set_input_data_text[2])/5) 
    + (float(lite_invoke_text[2])/5)
    
    return tf_cpu_lines, lite_cpu_lines

def write_2d_list_to_csv(data_2d, file_path):
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_2d)
        logger.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logger.info(f"Error occurred while writing to the file: {e}")

rows = [['model_idx', 'tf_cpu_line', 'lite_cpu_lite']]
for i in range(1, 13824 + 1):
    tf, lite = parse_profile(i)
    logger.info("{i} --> done")
    rows.append([i, tf, lite])
write_2d_list_to_csv(rows, "pi_cpu.csv")