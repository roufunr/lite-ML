
import logging
import os
import pandas as pd

home_path = os.path.expanduser('~')

def parse_profile(i):
    mem_profiler_file = f'{home_path}/mlp_profiler_result/pi/{i}_mem.txt'
    cpu_profiler_file = f'{home_path}/mlp_profiler_result/pi/{i}_cpu.txt'
    
    if(i == 610):
        print("Bug")
    
    with open(cpu_profiler_file, 'r') as file:
        cpu_profiler_lines = file.readlines()
    
    # print(cpu_profiler_lines)
    with open(mem_profiler_file, 'r') as file:
        mem_profiler_lines = file.readlines()
    
    # print(mem_profiler_lines)
    
    cpu_line_train_data_line = cpu_profiler_lines[16].split(" ")
    cpu_line_train_data_line = [x for x in cpu_line_train_data_line if x != '']

    cpu_line_test_data_line = cpu_profiler_lines[17].split(" ")
    cpu_line_test_data_line = [x for x in cpu_line_test_data_line if x != '']

    mem_train_data_line = mem_profiler_lines[14].split(" ")
    mem_train_data_line = [x for x in mem_train_data_line if x != '']

    mem_test_data_line = mem_profiler_lines[15].split(" ")
    mem_test_data_line = [x for x in mem_test_data_line if x != '']

    mem_addr_total_data_line = mem_profiler_lines[16].split(" ")
    mem_addr_total_data_line = [x for x in mem_addr_total_data_line if x != '']
    
    print(i)
    return [
        i,
        cpu_line_train_data_line[2],
        cpu_line_test_data_line[2],
        mem_train_data_line[3],
        mem_test_data_line[3],
        mem_addr_total_data_line[1]
    ]


rows = []
for i in range(1, 2268 + 1):
    row = parse_profile(i)
    rows.append(row)


df = pd.DataFrame(rows, columns=['idx', 'train(cpu_line)(nano sec)', 'test(cpu_line)(nano sec)', 'train(mem)(MiB)', 'test(mem)(MiB)', 'total_addr_space(mem)(MiB)'])

df.to_csv('cpu_mem_pi.csv', index=False)