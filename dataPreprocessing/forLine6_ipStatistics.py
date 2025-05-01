'''
功能: 统计每个src_ip的通信情况
思路: 读取CSV文件, 将timestamp字段转换为datetime类型, 统计每个src_ip的通信情况,保存统计结果到新的CSV文件
'''
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

def process_csv(file_path, output_dir):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将timestamp字段转换为datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 过滤掉 packet_size(B)、rtt(ms)、session_duration(s) 为0的数据
        df = df[(df['packet_size(B)'] > 0) & (df['rtt(ms)'] > 0) & (df['session_duration(s)'] > 0)]

        # 统计每个src_ip的通信情况
        result = df.groupby('src_ip').agg(
            dst_ip_count=('dst_ip', 'nunique'),
            start_time=('timestamp', 'min'),
            end_time=('timestamp', 'max'),
            mean_packet_size=('packet_size(B)', 'mean'),
            std_packet_size=('packet_size(B)', 'std'),
            max_packet_size=('packet_size(B)', 'max'),
            min_packet_size=('packet_size(B)', 'min'),
            mean_rtt=('rtt(ms)', 'mean'),
            std_rtt=('rtt(ms)', 'std'),
            max_rtt=('rtt(ms)', 'max'),
            min_rtt=('rtt(ms)', 'min'),
            mean_session_duration=('session_duration(s)', 'mean'),
            std_session_duration=('session_duration(s)', 'std'),
            max_session_duration=('session_duration(s)', 'max'),
            min_session_duration=('session_duration(s)', 'min')
        ).reset_index()

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 保存统计结果到新的CSV文件
        output_file = os.path.join(output_dir, file_name)
        result.to_csv(output_file, index=False)
        print(f"Processed {file_path} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(args):
    folder_path, base_output_dir = args
    output_dir = os.path.join(base_output_dir, os.path.basename(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc=f"Processing {folder_path}"):
        process_csv(csv_file, output_dir)

def main():
    base_dir = '/home/zhaohuan/桌面/byDirectionSort'
    base_output_dir = '/home/zhaohuan/桌面/byIpSta'
    group_numbers = [0, 1, 2, 3]

    for group_number in group_numbers:
        input_dir = f'{base_dir}-{group_number}'
        output_dir = f'{base_output_dir}-{group_number}'
        folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

        with Pool(processes=4) as pool:
            list(tqdm(pool.imap(process_folder, [(folder, output_dir) for folder in folders]), total=len(folders), desc=f"Processing group {group_number}"))

if __name__ == '__main__':
    main()