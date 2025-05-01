import os
import pandas as pd
from tqdm import tqdm

def process_csv(file_path, output_dir_ac, output_dir_ca, output_dir_bc, output_dir_cb):
    try:
        # 读取CSV文件，指定数据类型并设置low_memory=False
        dtype = {
            'column_8': str,  # Replace 'column_8' with the actual column name
            'column_13': str  # Replace 'column_13' with the actual column name
        }
        df = pd.read_csv(file_path, dtype=dtype, low_memory=False)

        # 分批处理数据
        batch_size = 10000
        for start in tqdm(range(0, len(df), batch_size), desc=f"Processing {os.path.basename(file_path)}"):
            batch_df = df.iloc[start:start + batch_size]

            # 筛选src_segment==A且dst_segment==C的数据
            df_ac = batch_df[(batch_df['src_segment'] == 'A') & (batch_df['dst_segment'] == 'C')]

            # 筛选src_segment==C且dst_segment==A的数据
            df_ca = batch_df[(batch_df['src_segment'] == 'C') & (batch_df['dst_segment'] == 'A')]

            # 筛选src_segment==B且dst_segment==C的数据
            df_bc = batch_df[(batch_df['src_segment'] == 'B') & (batch_df['dst_segment'] == 'C')]

            # 筛选src_segment==C且dst_segment==B的数据
            df_cb = batch_df[(batch_df['src_segment'] == 'C') & (batch_df['dst_segment'] == 'B')]

            # 获取文件名
            file_name = os.path.basename(file_path)

            # 处理A-C数据
            if not df_ac.empty:
                output_file_ac = os.path.join(output_dir_ac, file_name)
                if os.path.exists(output_file_ac):
                    df_ac.to_csv(output_file_ac, mode='a', header=False, index=False)
                else:
                    df_ac.to_csv(output_file_ac, mode='w', header=True, index=False)

            # 处理C-A数据
            if not df_ca.empty:
                output_file_ca = os.path.join(output_dir_ca, file_name)
                if os.path.exists(output_file_ca):
                    df_ca.to_csv(output_file_ca, mode='a', header=False, index=False)
                else:
                    df_ca.to_csv(output_file_ca, mode='w', header=True, index=False)

            # 处理B-C数据
            if not df_bc.empty:
                output_file_bc = os.path.join(output_dir_bc, file_name)
                if os.path.exists(output_file_bc):
                    df_bc.to_csv(output_file_bc, mode='a', header=False, index=False)
                else:
                    df_bc.to_csv(output_file_bc, mode='w', header=True, index=False)

            # 处理C-B数据
            if not df_cb.empty:
                output_file_cb = os.path.join(output_dir_cb, file_name)
                if os.path.exists(output_file_cb):
                    df_cb.to_csv(output_file_cb, mode='a', header=False, index=False)
                else:
                    df_cb.to_csv(output_file_cb, mode='w', header=True, index=False)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_group(group_number, base_input_dir, base_output_dir):
    input_dir = f'{base_input_dir}-{group_number}'
    output_dir_ac = f'{base_output_dir}{group_number}/A-C'
    output_dir_ca = f'{base_output_dir}{group_number}/C-A'
    output_dir_bc = f'{base_output_dir}{group_number}/B-C'
    output_dir_cb = f'{base_output_dir}{group_number}/C-B'

    # 创建输出目录
    os.makedirs(output_dir_ac, exist_ok=True)
    os.makedirs(output_dir_ca, exist_ok=True)
    os.makedirs(output_dir_bc, exist_ok=True)
    os.makedirs(output_dir_cb, exist_ok=True)

    # 获取所有CSV文件
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

    # 处理每个CSV文件
    for csv_file in tqdm(csv_files, desc=f"Processing CSV files for group {group_number}"):
        process_csv(csv_file, output_dir_ac, output_dir_ca, output_dir_bc, output_dir_cb)

def main():
    base_input_dir = '/home/zhaohuan/桌面/byDateClean'
    base_output_dir = '/home/zhaohuan/桌面/byDirection'

    group_numbers = [0, 1, 2, 3]

    for group_number in group_numbers:
        process_group(group_number, base_input_dir, base_output_dir)

if __name__ == '__main__':
    main()