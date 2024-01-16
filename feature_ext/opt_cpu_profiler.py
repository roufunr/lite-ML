import subprocess
import pyshark
import os
import sys
from line_profiler import LineProfiler

home_path = os.path.expanduser('~')
data_path = f"{home_path}/network_data"

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
    except Exception as e:
        print(f"An error occurred: {e}")
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
    first_12_packets = pyshark.FileCapture("temp.pcap")
    mat = pcap_to_matrix(first_12_packets)
    delete_temp_pcap()
    first_12_packets.clear()
    return mat


pcap_filename = sys.argv[1]

def generate_feature_matrix():
    pcap_filepath = f"{data_path}/{pcap_filename}"
    feature_matrix = generate_matrix(pcap_filepath)
    return feature_matrix

def profile_line(func, *args, **kwargs):
    profiler = LineProfiler()
    profiler.add_function(func)
    profiler.runcall(func, *args, **kwargs)
    profiler.print_stats()

profile_line(generate_feature_matrix)