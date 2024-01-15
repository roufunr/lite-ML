import subprocess
import time
import csv
import os
home_path = os.path.expanduser('~')
data_path = f"{home_path}/network_data"

def get_feature_packet_no_by_filter(file_path, filter, max_packet_analyze):
    try:
        
        tshark_command = ['tshark', '-r', file_path, '-Y', filter, '-c', str(max_packet_analyze), '-T', 'fields', '-e', 'frame.number']
        print(tshark_command)
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

        print(command)
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
        print(tshark_command)
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
        print(tshark_command)
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
        print(tshark_command)
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
        print(tshark_command)
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
        print(tshark_command)
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



smallest_file = f"{data_path}/IPMAC-10-7.pcap"
largest_file = f"{data_path}/IPMAC-18-10.pcap"


# print("Smallest file:")
# for i in range(1, 2):
#     start = time.time()
#     print(get_feature_matrix(smallest_file, 12))
#     end = time.time()
#     print(end - start)
#     # write to csv file
#     header = ["index", "time"]
#     data = [i, end-start]
#     with open('time_for_smallest.csv', 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(data)


print("Largest file:")
for i in range(1, 2):
    start = time.time()
    res =  get_feature_matrix(largest_file, 12)
    end = time.time()
    print(end - start)
    # # write to csv file
    # header = ["index", "time"]
    # data = [i, end-start]
    rows = []
    header = ['arp','llc','ip','icmp','icmpv6','eapol','tcp','udp','http','https','dhcp','bootp','ssdp','dns','mdns','ntp','padding','router_alert'	'size','raw_data','destination_ip_counter','source','destination']
    rows.append(header)

    for j in range(12):
        row = []
        for k in range(23):
            row.append(res[k][j])
        rows.append(row)

    with open('mat.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(rows)