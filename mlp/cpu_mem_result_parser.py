
import logging
import os
import pandas

home_path = os.path.expanduser('~')

def parse_profile(i):
    cpu_profiler_file = f'{home_path}/mlp_profiler_result/{i}_mem.txt'
    mem_profiler_file = f'{home_path}/mlp_profiler_result/{i}_cpu.txt'
    
    with open(cpu_profiler_file, 'r') as file:
        cpu_profiler_lines = file.readlines()
    
    print(cpu_profiler_lines)
    with open(mem_profiler_file, 'r') as file:
        mem_profiler_lines = file.readlines()
    
    print(mem_profiler_lines)
    # tf_cpu_line_text = profiler_lines[26].split(" ")
    # tf_cpu_line_text = [x for x in tf_cpu_line_text if x != '']
    
    # lite_allocate_tensor_text = profiler_lines[37].split(" ")
    # lite_allocate_tensor_text = [x for x in lite_allocate_tensor_text if x != '']
    
    # lite_get_input_tensor_text = profiler_lines[38].split(" ")
    # lite_get_input_tensor_text = [x for x in lite_get_input_tensor_text if x != '']
    
    # lite_set_input_data_text = profiler_lines[41].split(" ")
    # lite_set_input_data_text = [x for x in lite_set_input_data_text if x != '']
    
    # lite_invoke_text = profiler_lines[42].split(" ")
    # lite_invoke_text = [x for x in lite_invoke_text if x != '']
    
    # tf_cpu_lines = float(tf_cpu_line_text[2])/5
    
    # lite_cpu_lines = float(lite_allocate_tensor_text[2]) 
    # + float(lite_get_input_tensor_text[2]) 
    # + (float(lite_set_input_data_text[2])/5) 
    # + (float(lite_invoke_text[2])/5)
    
    # return tf_cpu_lines, lite_cpu_lines


rows = []
for i in range(1, 6318 + 1):
    row = parse_profile(i)
    rows.append(row)


# df = pandas.Dataframe(rows, columns=['idx', 'train(cpu_line)(nano sec)', 'test(cpu_line)(nano sec)', 'train(mem)(MiB)', 'test(mem)(MiB)', 'total_addr_space(mem)(MiB)'])