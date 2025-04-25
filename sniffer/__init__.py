import os

from scapy.utils import PcapReader

if __name__ == '__main__':
    file_path = '../static/test.pcap'
    try:
        with PcapReader(file_path) as pcap_reader:
            for packet in pcap_reader:
                print(packet.summary())
    except Exception as e:
        print(f"Error reading pcap file: {e}")