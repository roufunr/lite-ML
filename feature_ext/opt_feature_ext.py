import subprocess
import time
import csv
import os
import pyshark
import logging

home_path = os.path.expanduser('~')
data_path = f"{home_path}/network_data"
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/optcode.log',
                    filemode='a')

logger = logging.getLogger(__name__)

def get_feature_packet_no_by_filter(file_path, filter, max_packet_analyze):
    try:
        
        tshark_command = ['tshark', '-r', file_path, '-Y', filter, '-c', str(max_packet_analyze), '-T', 'fields', '-e', 'frame.number']
        # print(tshark_command)
        result = subprocess.run(tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        packet_numbers = result.stdout.strip().split('\n')

      
        result = [0] * max_packet_analyze

      
        for num in packet_numbers:
            if num:
                packet_index = int(num) - 1
                if packet_index < max_packet_analyze:
                    result[packet_index] = 1

        return result
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []

def get_protocol_in_packets(file, protocol, max_packet_analyze):
  
    try:
      
        command = [
            'tshark', '-r', file, '-Y', protocol, '-c', str(max_packet_analyze), '-T', 'fields', '-e', 'frame.number'
        ]

        # print(command)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        output = result.stdout.strip()

        protocol_packets = set(map(int, output.splitlines()))
        return [1 if i+1 in protocol_packets else 0 for i in range(max_packet_analyze)]
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []


def get_raw_data_matrix(file, max_packet_analyze):
    try:
        
        tshark_command = ['tshark', '-r', file, '-c', str(max_packet_analyze), '-T', 'fields', '-e', 'data']
        # print(tshark_command)
        result = subprocess.run(tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        raw_data_lines = result.stdout.split('\n')

        # Generate the result list
        result = [1 if line else 0 for line in raw_data_lines]
        return result
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []

def get_destination_ip_count(file, max_packet_analyze):
    try:
       
        tshark_command = ['tshark', '-r', file, '-c', str(max_packet_analyze),  '-T', 'fields', '-e', 'ip.dst']
        # print(tshark_command)
        result = subprocess.run(tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        destination_ips = result.stdout.split('\n')
        destination_ips.pop()

        # Count unique destination IPs
        unique_ips = set()
        ip_count = []
        for ip in destination_ips:
            if ip:
                unique_ips.add(ip)
            ip_count.append(len(unique_ips))
        return ip_count
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []

def get_src_port_class(file, max_packet_analyze):
    try:
        tshark_command = ['tshark', '-r', file, '-c', str(max_packet_analyze), '-T', 'fields', '-e', 'tcp.srcport', '-e', 'udp.srcport']
        # print(tshark_command)
        result = subprocess.run(tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        ports_raw = result.stdout.split('\n')
        ports_raw.pop()


     
        ports = []
        for port_pair in ports_raw:
            tcp_port, udp_port = (port_pair.split('\t') + [''])[:2]
        
          
            src_port = tcp_port or udp_port  
            if src_port == '':
                port_class = 0
            else:
                src_port = int(src_port)
                if 0 <= src_port <= 1023:
                    port_class = 1
                elif 1024 <= src_port <= 49151:
                    port_class = 2
                elif 49152 <= src_port <= 65535:
                    port_class = 3
            ports.append(port_class)

        return ports
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []


def get_dst_port_class(file, max_packet_analyze):
    try:
        
        tshark_command = ['tshark', '-r', file, '-c', str(max_packet_analyze), '-T', 'fields', '-e', 'tcp.dstport', '-e', 'udp.dstport']
        # print(tshark_command)
        result = subprocess.run(tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        ports_raw = result.stdout.split('\n')
        ports_raw.pop()

        
        ports = []
        for port_pair in ports_raw:
            tcp_port, udp_port = (port_pair.split('\t') + [''])[:2]

        
            dst_port = tcp_port or udp_port 
            if dst_port == '':
                port_class = 0
            else:
                dst_port = int(dst_port)
                if 0 <= dst_port <= 1023:
                    port_class = 1
                elif 1024 <= dst_port <= 49151:
                    port_class = 2
                elif 49152 <= dst_port <= 65535:
                    port_class = 3
            ports.append(port_class)

        return ports
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []


def get_packet_size_matrix(file, max_packet_analyze):
    try:
        
        tshark_command = ['tshark', '-r', file, '-T', 'fields', '-e', 'frame.len']
        # print(tshark_command)
        result = subprocess.run(tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        packet_sizes = result.stdout.strip().split('\n')

        # Convert packet sizes to integers and return the required number
        return [int(size) for size in packet_sizes[:max_packet_analyze] if size]
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []

row_header = ["arp", "llc", "ip", "icmp", "icmpv6", "eapol", "tcp", "udp", "http", "https", "dhcp", "bootp", "ssdp", "dns", "mdns", "ntp", "padding", "router_alert", "size", "raw_data", "destination_ip_counter", "source", "destination"]
def get_feature_matrix(file, max_packet_analyze):
    # ARP
    arp_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='arp')
    llc_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='llc')
    ip_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='ip')
    icmp_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='icmp')
    icmpv6_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='icmpv6')
    eapol_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='eapol')
    tcp_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='tcp')
    udp_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='udp')
    http_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='http')
    https_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='ssl')
    dhcp_matrix = get_feature_packet_no_by_filter(file_path=file, filter="udp.port eq 67 or udp.port eq 68", max_packet_analyze=max_packet_analyze)
    bootp_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='bootp')
    ssdp_matrix = get_feature_packet_no_by_filter(file_path=file, filter="udp.port eq 1900", max_packet_analyze=max_packet_analyze)
    dns_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='dns')
    mdns_matrix = get_feature_packet_no_by_filter(file_path=file, filter="dns and udp.port eq 5353", max_packet_analyze=max_packet_analyze)
    ntp_matrix = get_protocol_in_packets(file=file, max_packet_analyze=max_packet_analyze, protocol='ntp')
    ip_option_paddding_matrix = get_feature_packet_no_by_filter(file_path=file, filter="ip.opt.padding", max_packet_analyze=max_packet_analyze)
    ip_option_router_alert_matrix = get_feature_packet_no_by_filter(file_path=file, filter="ip.opt.ra", max_packet_analyze=max_packet_analyze)
    packet_size_matrix = get_packet_size_matrix(file=file, max_packet_analyze=max_packet_analyze)
    raw_data_matrix = get_raw_data_matrix(file=file, max_packet_analyze=max_packet_analyze)
    destination_ip_count = get_destination_ip_count(file=file, max_packet_analyze=max_packet_analyze)
    src_port_class = get_src_port_class(file=file, max_packet_analyze=max_packet_analyze)
    dst_port_class = get_dst_port_class(file=file, max_packet_analyze=max_packet_analyze)

     

    return [
        arp_matrix,
        llc_matrix,
        ip_matrix,
        icmp_matrix,
        icmpv6_matrix,
        eapol_matrix,
        tcp_matrix,
        udp_matrix,
        http_matrix,
        https_matrix,
        dhcp_matrix,
        bootp_matrix,
        ssdp_matrix,
        dns_matrix,
        mdns_matrix,
        ntp_matrix,
        ip_option_paddding_matrix,
        ip_option_router_alert_matrix,
        packet_size_matrix,
        raw_data_matrix,
        destination_ip_count,
        src_port_class,
        dst_port_class
    ]


def pcap_to_matrix(packets):
    matrix = [[0 for j in range(23)] for i in range(12)]
    destination_ips = set()
    try:
        for idx, packet in enumerate(packets):
            if idx == 11:
                print(1)
            protocols = packet.frame_info.protocols.split(":")
            if "arp" in protocols:
                # arp
                matrix[idx][0] = 1
            if "llc" in protocols:
                # llc
                matrix[idx][1] = 1
            if "ip" in protocols:
                # ip
                matrix[idx][2] = 1
            if "icmp" in protocols:
                # icmp
                matrix[idx][3] = 1
            if "icmpv6" in protocols:
                # icmpv6
                matrix[idx][4] = 1
            if "eapol" in protocols:
                # eapol
                matrix[idx][5] = 1
            if "tcp" in protocols:
                # tcp
                matrix[idx][6] = 1
            if "DHCP" in packet:
                # dhcp
                matrix[idx][10] = 1
            if "udp" in protocols:
                # udp
                matrix[idx][7] = 1
                if hasattr(packet, "udp"):
                    ports = [packet.udp.srcport, packet.udp.dstport]
                    if '67' in ports or '68' in ports:    
                        # bootp
                        matrix[idx][11] = 1
                    if '1900' in ports:
                        # ssdp
                        matrix[idx][12] = 1
            if "http" in protocols:
                # http
                matrix[idx][8] = 1
            if 'TCP' in packet and 'TLS' in packet:
                # https
                matrix[idx][9] = 1
            if "dns" in protocols:
                # dns
                matrix[idx][13] = 1
            if "dns" in protocols and "udp" in protocols:
                ports = [packet.udp.srcport, packet.udp.dstport]
                if '5353' in ports:
                    # mdns
                    matrix[idx][14] = 1
            if "ntp" in protocols:
                # ntp
                matrix[idx][15] = 1
            if 'IP' in packet:
                if hasattr(packet.ip, 'opt_padding'):
                    # padding
                    matrix[idx][16] = 1
                if hasattr(packet.ip, 'opt_ra'):
                    # router alert
                    matrix[idx][17] = 1
            
            
            # size
            matrix[idx][18] = int(packet.frame_info.len)
            
            # raw data present
            matrix[idx][19] = 0 if packet.layers[-1].get_field_value('data') == None else 1
            
            if 'IP' in packet:
                dst_ip = packet.ip.dst
                if dst_ip:
                    destination_ips.add(dst_ip)
            # destination ip count
            matrix[idx][20] = len(destination_ips)
            
            if 'IP' in packet and ('tcp' in packet or 'udp' in packet):
                src_port = packet[packet.transport_layer].srcport
                dst_port = packet[packet.transport_layer].dstport
                if src_port:
                    src_port = int(src_port)
                    if 0 <= src_port <= 1023:
                        port_class = 1
                    elif 1024 <= src_port <= 49151:
                        port_class = 2
                    elif 49152 <= src_port <= 65535:
                        port_class = 3
                    else:
                        port_class = 0
                    
                    # source port class
                    matrix[idx][21] = port_class
                if dst_port:
                    dst_port = int(dst_port)
                    if 0 <= dst_port <= 1023:
                        port_class = 1
                    elif 1024 <= dst_port <= 49151:
                        port_class = 2
                    elif 49152 <= dst_port <= 65535:
                        port_class = 3
                    else:
                        port_class = 0
                    # destination port class
                    matrix[idx][22] = port_class
    except:
        print("exception")
    return matrix
    
def chop_down_pcap(pcap_file_path):
    command = ["tshark", "-r", pcap_file_path, "-w", "temp.pcap", "-c", "12"]
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

def delete_temp_pcap():
    file_path = "temp.pcap"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    else:
        print(f"File '{file_path}' not found.")
       
def generate_matrix(pcap_file_path):
    chop_down_pcap(pcap_file_path)
    cap = cap = pyshark.FileCapture("temp.pcap")
    first_12_packets = [packet for idx, packet in enumerate(cap) if idx < 12]
    mat = pcap_to_matrix(first_12_packets)
    delete_temp_pcap()
    return mat 


def isMatched(res1, res2):
    for i in range(12):
        for j in range(23):
            if res1[j][i] != res2[i][j]:
                logger.info(f"{i}, {j}, {res1[j][i]}, {res2[i][j]}")
                return False
    return True

def get_device_sample_map():
    pcap_data_path = f'{home_path}/network_data'
    files = os.listdir(pcap_data_path)
    device_sample_map = {i: [] for i in range(1, 32)}
    for filename in files:
        pieces = filename.split("-")
        device_id = pieces[1]
        sample_id = pieces[2].split(".")[0]
        # print(device_id, sample_id)
        device_sample_map[int(device_id)].append(int(sample_id))

    for i in range(1, 32):
        device_sample_map[i] = sorted(device_sample_map[i])
        
    return device_sample_map

device_sample_map = get_device_sample_map()

i = 1
rows = [['idx', 'device', 'sample', 'result_matched', 'old_time(ms)', 'new_time(ms)', 'diff_time(ms)', 'improved (%)']]
for device in range(23, 32):
    for sample in device_sample_map[device]:
        filepath = f"{home_path}/network_data/IPMAC-{device}-{sample}.pcap"
        start = time.time() * 1000
        res1 =  get_feature_matrix(filepath, 12)
        end = time.time() * 1000
        old_time = end - start 
        
        start = time.time() * 1000
        res2 =  generate_matrix(filepath)
        end = time.time() * 1000
        new_time = end - start
        
        matched = isMatched(res1, res2)

        diff = old_time - new_time
        improved_percentage = round((diff/old_time) * 100)
        rows.append([i,device, sample, matched, old_time, new_time, diff, improved_percentage])
        logger.info(f"idx: {i}, device_id: {device}, sample_id: {sample}, matched: {matched}, old_time: {old_time}, new_time: {new_time}, time-difference: {diff}, improved: {improved_percentage}% --> DONE")
        i += 1
        
with open('comparison.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(rows)


# filepath = f"{home_path}/network_data/IPMAC-{18}-{3}.pcap"
# start = time.time() * 1000
# res1 =  get_feature_matrix(filepath, 12)
# end = time.time() * 1000
# old_time = end - start 

# start = time.time() * 1000
# res2 =  generate_matrix(filepath)
# end = time.time() * 1000
# new_time = end - start

# matched = isMatched(res1, res2)

# diff = old_time - new_time
# improved_percentage = round((diff/old_time) * 100)