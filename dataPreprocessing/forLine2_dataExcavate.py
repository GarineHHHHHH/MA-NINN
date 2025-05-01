'''
根据pcap组成,分别提取结尾为0、1、2、3的四组数据,并分别存放到byDateCSV-0、byDateCSV-1、byDateCSV-2、byDateCSV-3四个文件夹中。
统计不同日期下timestamp,src_ip,src_segment,dst_ip,dst_segment,ttl(hop_count),protocol,src_port,dst_port,tcp_flags,window_size(B),packet_size(B),rtt(ms),session_duration(s),date
'''
import os
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm

def extract_features(pcap_file, output_dir):
    try:
        # 检查目标目录下是否存在与流量包同名的CSV文件
        csv_file = os.path.join(output_dir, os.path.basename(os.path.splitext(pcap_file)[0]) + '.csv')
        if os.path.exists(csv_file):
            print(f"File {csv_file} already exists. Skipping {pcap_file}.")
            return

        packets = rdpcap(pcap_file)
        if not packets:
            print(f"No packets found in {pcap_file}")
            return

        features = []
        tcp_sessions = {}

        for packet in packets:
            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                ttl = packet[IP].ttl
                protocol = packet[IP].proto
                timestamp = datetime.fromtimestamp(float(packet.time)).strftime('%Y-%m-%d %H:%M:%S')

                # Determine network segment
                if src_ip.startswith('10.48.'):
                    src_segment = 'A'
                elif src_ip.startswith('10.123.'):
                    src_segment = 'B'
                else:
                    src_segment = 'C'

                if dst_ip.startswith('10.48.'):
                    dst_segment = 'A'
                elif dst_ip.startswith('10.123.'):
                    dst_segment = 'B'
                else:
                    dst_segment = 'C'

                # Transport layer features
                src_port = dst_port = tcp_flags = window_size = None
                if TCP in packet:
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    tcp_flags = packet[TCP].flags
                    window_size = packet[TCP].window

                    # Track TCP sessions
                    session_key = (src_ip, dst_ip, src_port, dst_port)
                    if session_key not in tcp_sessions:
                        tcp_sessions[session_key] = {
                            'timestamps': [],
                            'start_time': packet.time,
                            'end_time': packet.time
                        }
                    tcp_sessions[session_key]['timestamps'].append(packet.time)
                    tcp_sessions[session_key]['end_time'] = packet.time

                elif UDP in packet:
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport

                # Flow statistics
                packet_size = len(packet)
                rtt = None  # RTT will be calculated later

                features.append([
                    timestamp, src_ip, src_segment, dst_ip, dst_segment, ttl, protocol,
                    src_port, dst_port, tcp_flags, window_size, packet_size, rtt
                ])

        # Calculate RTT and session duration for TCP sessions
        for session_key, session_data in tcp_sessions.items():
            timestamps = session_data['timestamps']
            if len(timestamps) > 1:
                rtt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
            else:
                rtt = 0
            session_duration = session_data['end_time'] - session_data['start_time']

            # Update features with calculated RTT and session duration
            for feature in features:
                if (feature[1], feature[3], feature[7], feature[8]) == session_key:
                    feature[-1] = rtt
                    feature.append(session_duration)

        # Define CSV columns
        columns = [
            'timestamp', 'src_ip', 'src_segment', 'dst_ip', 'dst_segment', 'ttl(hop_count)', 'protocol',
            'src_port', 'dst_port', 'tcp_flags', 'window_size(B)', 'packet_size(B)', 'rtt(ms)', 'session_duration(s)'
        ]

        # Save features to CSV
        df = pd.DataFrame(features, columns=columns)
        df.to_csv(csv_file, index=False)
        print(f"Processed {pcap_file} and saved to {csv_file}")

    except Exception as e:
        print(f"Error processing {pcap_file}: {e}")

def process_folder(args):
    folder_path, group_number = args
    output_dir = os.path.join(f'/home/zhaohuan/桌面/byDateCSV-{group_number}', os.path.basename(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    pcap_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(f'{group_number}.pcap')]
    for pcap_file in tqdm(pcap_files, desc=f"Processing {folder_path}"):
        extract_features(pcap_file, output_dir)

def main():
    base_dir = '/home/zhaohuan/桌面/byDate'
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    group_numbers = [0, 1, 2, 3]

    # 先处理 group_number 为 0, 1, 2 的任务
    for group_number in group_numbers[:-1]:
        tasks = [(folder, group_number) for folder in folders]
        with Pool(processes=41) as pool:
            list(tqdm(pool.imap(process_folder, tasks), total=len(tasks), desc=f"Processing group {group_number}"))

    # 最后处理 group_number 为 3 的任务
    tasks = [(folder, 3) for folder in folders]
    with Pool(processes=41) as pool:
        list(tqdm(pool.imap(process_folder, tasks), total=len(tasks), desc="Processing group 3"))

if __name__ == '__main__':
    main()
