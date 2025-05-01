'''
对byDateCSV文件夹下的所有csv文件进行数据清洗,
删除存在字段缺失的数据,
并根据timestamp字段将数据分组,
将每个日期分组的数据保存到byDateClean文件夹下的对应日期的csv文件中。
'''

import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

def clean_and_split_csv(file_path, output_dir):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 删除存在字段缺失的数据
        df.dropna(inplace=True)

        # 根据timestamp字段将数据分组
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        # 遍历每个日期分组
        for date, group in df.groupby('date'):
            output_file = os.path.join(output_dir, f"{date}.csv")
            if os.path.exists(output_file):
                group.to_csv(output_file, mode='a', header=False, index=False)
            else:
                group.to_csv(output_file, mode='w', header=True, index=False)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(args):
    folder_path, group_number = args
    output_dir = f'/home/zhaohuan/桌面/byDateClean-{group_number}'
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc=f"Processing {folder_path}"):
        clean_and_split_csv(csv_file, output_dir)

def main():
    base_dir = '/home/zhaohuan/桌面/byDateCSV'
    group_numbers = [0, 1, 2, 3]

    for group_number in group_numbers:
        folders = [os.path.join(f'{base_dir}-{group_number}', d) for d in os.listdir(f'{base_dir}-{group_number}') if os.path.isdir(os.path.join(f'{base_dir}-{group_number}', d))]
        tasks = [(folder, group_number) for folder in folders]

        with Pool(processes=41) as pool:
            list(tqdm(pool.imap(process_folder, tasks), total=len(tasks), desc=f"Processing group {group_number}"))

if __name__ == '__main__':
    main()