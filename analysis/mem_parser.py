import logging
import os
import csv 

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/mem_analysis.log',
                    filemode='a')

logger = logging.getLogger(__name__)

def parse_memory_profile(i):
    tf = f'{home_path}/mem_utilization/orin/{i}_tf.txt'
    lite = f'{home_path}/mem_utilization/orin/{i}_lite.txt'
    with open(tf, 'r') as file:
        tf_lines = file.readlines()
    with open(lite, 'r') as file:
        lite_lines = file.readlines()
    
    tf_mem_req_to_load_model = float(tf_lines[21].split(" ")[11])
    tf_mem_req_to_infer_model = float(tf_lines[26].split(" ")[11])
    tf_total_mem = float(tf_lines[26].split(" ")[7])
    
    lite_mem_req_to_load_model = float(lite_lines[6].split(" ")[15]) + float(lite_lines[7].split(" ")[15])
    lite_mem_req_to_infer_model = float(lite_lines[13].split(" ")[15])
    lite_total_mem = float(lite_lines[13].split(" ")[8])
    logger.info(f"{i} -- > DONE")
    
    data = [
        i, 
        tf_mem_req_to_load_model, 
        lite_mem_req_to_load_model,
        tf_mem_req_to_infer_model, 
        lite_mem_req_to_infer_model,
        tf_total_mem,
        lite_total_mem
    ]
    return data

def write_2d_list_to_csv(data_2d, file_path):
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_2d)
        logger.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logger.info(f"Error occurred while writing to the file: {e}")

rows = [['model_idx', 'tf_mem_req_to_load_model', 'lite_mem_req_to_load_model', 'tf_mem_req_to_infer_model', 'lite_mem_req_to_infer_model', 'tf_total_mem', 'lite_total_mem',]]
for i in range(1, 13825):
    data = parse_memory_profile(i)
    rows.append(data)

write_2d_list_to_csv(rows, "orin_mem.csv")

