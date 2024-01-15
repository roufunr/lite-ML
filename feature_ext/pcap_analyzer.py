import pyshark

def analyze_filter(packets, display_filter, tshark_filter):
    print(f"Filter: {display_filter}")
    
    for idx, packet in enumerate(packets):
        print(f"{idx + 1}\t{packet.frame_info.number}")

    print("-----------------------------")

def main():
    pcap_file = "/home/rouf-linux/network_data/IPMAC-18-10.pcap"

    # Read the first 12 packets
    cap = pyshark.FileCapture(pcap_file)
    first_12_packets = [packet for idx, packet in enumerate(cap) if idx < 12]

    # Analyze filters for the first 12 packets
    analyze_filter(first_12_packets, "ARP", "arp")
    analyze_filter(first_12_packets, "LLC", "llc")
    analyze_filter(first_12_packets, "IP", "ip")
    analyze_filter(first_12_packets, "ICMP", "icmp")
    analyze_filter(first_12_packets, "ICMPv6", "icmpv6")
    analyze_filter(first_12_packets, "EAPOL", "eapol")
    analyze_filter(first_12_packets, "TCP", "tcp")
    analyze_filter(first_12_packets, "UDP", "udp")
    analyze_filter(first_12_packets, "HTTP", "http")
    analyze_filter(first_12_packets, "SSL", "ssl")
    analyze_filter(first_12_packets, "BOOTP/DHCP", "(udp.port eq 67 or udp.port eq 68)")
    analyze_filter(first_12_packets, "SSDP", "(udp.port eq 1900)")
    analyze_filter(first_12_packets, "DNS", "dns")
    analyze_filter(first_12_packets, "mDNS", "(dns and udp.port eq 5353)")
    analyze_filter(first_12_packets, "NTP", "ntp")
    analyze_filter(first_12_packets, "IP Option Padding", "ip.opt.padding")
    analyze_filter(first_12_packets, "IP Option Router Alert", "ip.opt.ra")

    # Additional fields for the first 12 packets
    print("Frame Length")
    for idx, packet in enumerate(first_12_packets):
        print(f"{idx + 1}\t{packet.length}")

    print("\nData")
    for idx, packet in enumerate(first_12_packets):
        print(f"{idx + 1}\t{packet.data}")

    print("\nIP Destination")
    for idx, packet in enumerate(first_12_packets):
        if 'IP' in packet:
            print(f"{idx + 1}\t{packet.ip.dst}")

    print("\nSource and Destination Ports")
    for idx, packet in enumerate(first_12_packets):
        if 'TCP' in packet:
            print(f"{idx + 1}\t{packet.tcp.srcport}\t{packet.tcp.dstport}")
        elif 'UDP' in packet:
            print(f"{idx + 1}\t{packet.udp.srcport}\t{packet.udp.dstport}")

if __name__ == "__main__":
    main()
