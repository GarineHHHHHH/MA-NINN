import os
import pandas as pd
from tqdm import tqdm

def process_csv(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 删除time列中的日期部分，只保留时分秒
        df['time'] = pd.to_datetime(df['time']).dt.strftime('%H:%M:%S')

        # 保存修改后的数据到原CSV文件
        df.to_csv(file_path, index=False)
        print(f"Processed {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(folder_path):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc=f"Processing {folder_path}"):
        process_csv(csv_file)

def main():
    base_dir = 'forLines/byDayLineConstant-(min)'
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for folder in folders:
        process_folder(folder)

if __name__ == '__main__':
    main()