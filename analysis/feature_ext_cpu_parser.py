import logging
import os
import csv 

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
data_path = f"{home_path}/feature_ext_profiler_result_new"

os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/mem_analysis.log',
                    filemode='a')

logger = logging.getLogger(__name__)

def parse_cpu_profile(i, j):
    pc = f'{data_path}/pc/{i}_{j}_cpu.txt'
    orin = f'{data_path}/orin/{i}_{j}_cpu.txt'
    pi = f'{data_path}/pi/{i}_{j}_cpu.txt'
    
    with open(pc, 'r') as file:
        pc_lines = file.readlines()
    
    with open(orin, 'r') as file:
        orin_lines = file.readlines()
    
    with open(pi, 'r') as file:
        pi_lines = file.readlines()
        
    pc_lines_text = pc_lines[13].split(" ")
    pc_lines_text = [x for x in pc_lines_text if x != '']
    
    orin_lines_text = orin_lines[13].split(" ")
    orin_lines_text = [x for x in orin_lines_text if x != '']
    
    pi_lines_text = pi_lines[13].split(" ")
    pi_lines_text = [x for x in pi_lines_text if x != '']
    
    return (float(pc_lines_text[2]), float(orin_lines_text[2]), float(pi_lines_text[2]))

def write_2d_list_to_csv(data_2d, file_path):
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_2d)
        logger.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logger.info(f"Error occurred while writing to the file: {e}")

rows = [['idx', 'device', 'sample', 'cpu_line (pc)', 'cpu_line (orin)', 'cpu_line (pi)']]


def get_device_sample_map():
    pcap_data_path = f'{home_path}/network_data'
    files = os.listdir(pcap_data_path)
    device_sample_map = {i: [] for i in range(1, 32)}
    for filename in files:
        pieces = filename.split("-")
        device_id = pieces[1]
        sample_id = pieces[2].split(".")[0]
        print(device_id, sample_id)
        device_sample_map[int(device_id)].append(int(sample_id))

    for i in range(1, 32):
        device_sample_map[i] = sorted(device_sample_map[i])
        
    return device_sample_map

device_sample_map = get_device_sample_map()

i = 1
for device in range(1, 32):
    for sample in device_sample_map[device]:
        pc, orin, pi = parse_cpu_profile(device, sample)
        row = [i, device, sample, pc, orin, pi]
        rows.append(row)
        i += 1

write_2d_list_to_csv(rows, "new_feature_ext_cpuline.csv")