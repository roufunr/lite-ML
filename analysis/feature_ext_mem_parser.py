import logging
import os
import csv 

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
data_path = f"{home_path}/feature_ext_profiler_result"

os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/mem_analysis.log',
                    filemode='a')

logger = logging.getLogger(__name__)

def parse_mem_profile(i, j):
    pc = f'{data_path}/pc/{i}_{j}_mem.txt'
    orin = f'{data_path}/orin/{i}_{j}_mem.txt'
    pi = f'{data_path}/pi/{i}_{j}_mem.txt'
    
    with open(pc, 'r') as file:
        pc_mem = file.readlines()
    
    with open(orin, 'r') as file:
        orin_mem = file.readlines()
    
    with open(pi, 'r') as file:
        pi_mem = file.readlines()
        
    pc_mem_text = pc_mem[8].split(" ")
    pc_mem_text = [x for x in pc_mem_text if x != '']
    pc_mem_inc_text = pc_mem[7].split(" ")
    pc_mem_inc_text = [x for x in pc_mem_inc_text if x != '']
    
    orin_mem_text = orin_mem[8].split(" ")
    orin_mem_text = [x for x in orin_mem_text if x != '']
    orin_mem_inc_text = orin_mem[7].split(" ")
    orin_mem_inc_text = [x for x in orin_mem_inc_text if x != '']
    
    pi_mem_text = pi_mem[8].split(" ")
    pi_mem_text = [x for x in pi_mem_text if x != '']
    pi_mem_inc_text = pi_mem[7].split(" ")
    pi_mem_inc_text = [x for x in pi_mem_inc_text if x != '']
    
    return (
        float(pc_mem_text[1]),
        float(orin_mem_text[1]),
        float(pi_mem_text[1]),
        float(pc_mem_inc_text[3]),
        float(orin_mem_inc_text[3]),
        float(pi_mem_inc_text[3])
    )
    
    

def write_2d_list_to_csv(data_2d, file_path):
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_2d)
        logger.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logger.info(f"Error occurred while writing to the file: {e}")

rows = [['idx', 'device', 'sample', 'total_mem (pc)', 'total_mem (orin)', 'total_mem (pi)', 'mem_increment (pc)', 'mem_increment (orin)', 'mem_increment (pi)']]


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

# print(parse_mem_profile(1, 1))

i = 1
for device in range(1, 32):
    for sample in device_sample_map[device]:
        pc, orin, pi, pc_i, orin_i, pi_i = parse_mem_profile(device, sample)
        row = [i, device, sample, pc, orin, pi, pc_i, orin_i, pi_i]
        rows.append(row)
        i += 1

write_2d_list_to_csv(rows, "feature_ext_mem.csv")