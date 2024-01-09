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
    with open(profiler_file, 'r') as file:
        profiler_lines = file.readlines()
    
    
    tf_cpu_lines = (float(profiler_lines[26].split(" ")[15]))/5
    
    lite_cpu_lines = float(profiler_lines[38].split(" ")[19]) + float(profiler_lines[38].split(" ")[19]) + float(profiler_lines[41].split(" ")[22]) + float(profiler_lines[42].split(" ")[22])
    
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
for i in range(1, 1 + 1):
    tf, lite = parse_profile(i)
    rows.append([i, tf, lite])
write_2d_list_to_csv(rows, "pc_cpu.csv")

