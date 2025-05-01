'''
对文件夹下的所有CSV文件按照timestamp字段进行排序,并保存到新的CSV文件中,保留原目录结构
'''
import os
import pandas as pd
from tqdm import tqdm

def sort_csv(file_path, output_dir):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 根据timestamp字段对数据进行排序
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 保存排序后的数据到新的CSV文件
        output_file = os.path.join(output_dir, file_name)
        df.to_csv(output_file, index=False)
        print(f"Processed {file_path} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(folder_path, base_output_dir):
    output_dir = os.path.join(base_output_dir, os.path.basename(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc=f"Processing {folder_path}"):
        sort_csv(csv_file, output_dir)

def main():
    base_dir = '/home/zhaohuan/桌面/byDirection'
    base_output_dir = '/home/zhaohuan/桌面/byDirectionSort'
    group_numbers = [0, 1, 2, 3]

    for group_number in group_numbers:
        input_dir = f'{base_dir}-{group_number}'
        output_dir = f'{base_output_dir}-{group_number}'
        folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

        for folder in tqdm(folders, desc=f"Processing folders for group {group_number}"):
            process_folder(folder, output_dir)

if __name__ == '__main__':
    main()