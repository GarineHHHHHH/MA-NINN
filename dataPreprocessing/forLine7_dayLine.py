'''
对byDirectionSort文件夹下的所有CSV文件进行数据处理,
生成time_granularity = s 和 min的两类基线,
分别存储在byDayLine-{0-3}-({time_granularity})中
'''

import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

def min_nonzero(series):
    non_zero_series = series[series != 0]
    if not non_zero_series.empty:
        return non_zero_series.min()
    else:
        return 0

def process_csv(file_path, output_dir, time_granularity):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将timestamp字段转换为datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if time_granularity == 's':
            # 将timestamp字段按秒进行聚合
            df['time'] = df['timestamp'].dt.floor('s')
        elif time_granularity == 'min':
            # 将timestamp字段按分钟进行聚合
            df['time'] = df['timestamp'].dt.floor('min')
        else:
            raise ValueError("Invalid time granularity. Use 's' for seconds or 'min' for minutes.")

        # 统计每天的基线数据
        baseline_data = df.groupby('time').agg(
            src_ip_count=('src_ip', 'nunique'),
            dst_ip_count=('dst_ip', 'nunique'),
            mean_window_size=('window_size(B)', 'mean'),
            max_window_size=('window_size(B)', 'max'),
            min_window_size=('window_size(B)', min_nonzero),
            mean_packet_size=('packet_size(B)', 'mean'),
            std_packet_size=('packet_size(B)', 'std'),
            max_packet_size=('packet_size(B)', 'max'),
            min_packet_size=('packet_size(B)', min_nonzero),
            mean_rtt=('rtt(ms)', 'mean'),
            std_rtt=('rtt(ms)', 'std'),
            max_rtt=('rtt(ms)', 'max'),
            min_rtt=('rtt(ms)', min_nonzero),
            mean_session_duration=('session_duration(s)', 'mean'),
            std_session_duration=('session_duration(s)', 'std'),
            max_session_duration=('session_duration(s)', 'max'),
            min_session_duration=('session_duration(s)', min_nonzero)
        ).reset_index()

        # 删除包含零值的行
        baseline_data.replace(0, pd.NA, inplace=True)
        baseline_data.dropna(inplace=True)

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 保存统计结果到新的CSV文件
        output_file = os.path.join(output_dir, file_name)
        baseline_data.to_csv(output_file, index=False)
        print(f"Processed {file_path} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(args):
    folder_path, base_output_dir, time_granularity = args
    output_dir = os.path.join(base_output_dir, os.path.basename(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc=f"Processing {folder_path}"):
        process_csv(csv_file, output_dir, time_granularity)

def main():
    base_dir = '/home/zhaohuan/桌面/byDirectionSort'
    time_granularities = ['s', 'min']
    base_output_dir_template = '/home/zhaohuan/桌面/byDayLine-{group_number}-({time_granularity})'
    group_numbers = [0, 1, 2, 3]

    for group_number in group_numbers:
        for time_granularity in time_granularities:
            base_output_dir = base_output_dir_template.format(group_number=group_number, time_granularity=time_granularity)
            input_dir = f'{base_dir}-{group_number}'
            folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

            with Pool(processes=4) as pool:
                list(tqdm(pool.imap(process_folder, [(folder, base_output_dir, time_granularity) for folder in folders]), total=len(folders), desc=f"Processing group {group_number} with granularity {time_granularity}"))

if __name__ == '__main__':
    main()