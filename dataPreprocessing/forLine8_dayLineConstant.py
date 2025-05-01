import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
def interpolate_data(file_path, output_dir, time_granularity):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将time字段转换为datetime类型
        df['time'] = pd.to_datetime(df['time'])

        # 设置时间索引
        df.set_index('time', inplace=True)

        # 生成完整的时间索引
        if time_granularity == 's':
            full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='S')
        elif time_granularity == 'min':
            full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')
        else:
            raise ValueError("Invalid time granularity. Use 's' for seconds or 'min' for minutes.")

        # 重建DataFrame，使用完整的时间索引
        df = df.reindex(full_time_index)

        # 插值补全空缺数据
        df.interpolate(method='time', inplace=True)

        # 填充前后空缺数据
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        # 重置索引
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'time'}, inplace=True)

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 保存插值后的数据到新的CSV文件
        output_file = os.path.join(output_dir, file_name)
        df.to_csv(output_file, index=False)
        print(f"Processed {file_path} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def interpolate_data_with_lstm(file_path, output_dir, time_granularity, time_steps=10):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将time字段转换为datetime类型
        df['time'] = pd.to_datetime(df['time'])

        # 设置时间索引
        df.set_index('time', inplace=True)

        # 生成完整的时间索引
        if time_granularity == 's':
            full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='S')
        elif time_granularity == 'min':
            full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')
        else:
            raise ValueError("Invalid time granularity. Use 's' for seconds or 'min' for minutes.")

        # 重建DataFrame，使用完整的时间索引
        df = df.reindex(full_time_index)

        # 标记空缺数据
        missing_mask = df.isnull()

        # 填充空缺数据为 0（仅用于训练 LSTM）
        df.fillna(0, inplace=True)

        # 准备 LSTM 输入数据
        values = df.values  # 转换为 numpy 数组
        X, y = [], []
        for i in range(len(values) - time_steps):
            X.append(values[i:i + time_steps])
            y.append(values[i + time_steps])
        X, y = np.array(X), np.array(y)

        # 构建 LSTM 模型
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(time_steps, X.shape[2])))
        model.add(Dense(X.shape[2]))  # 输出与特征数量一致
        model.compile(optimizer='adam', loss='mse')

        # 训练 LSTM 模型
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)

        # 使用 LSTM 模型预测空缺数据
        for i in range(len(df)):
            if missing_mask.iloc[i].any():  # 如果当前行有空缺值
                start_idx = max(0, i - time_steps)
                end_idx = i
                input_data = df.iloc[start_idx:end_idx].values
                if len(input_data) < time_steps:  # 如果数据不足 time_steps，则填充 0
                    input_data = np.pad(input_data, ((time_steps - len(input_data), 0), (0, 0)), mode='constant')
                input_data = input_data.reshape(1, time_steps, -1)
                predicted = model.predict(input_data)
                df.iloc[i] = predicted  # 用预测值填充空缺数据

        # 重置索引
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'time'}, inplace=True)

        # 获取文件名
        file_name = os.path.basename(file_path)

        # 保存插值后的数据到新的CSV文件
        output_file = os.path.join(output_dir, file_name)
        df.to_csv(output_file, index=False)
        print(f"Processed {file_path} and saved to {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder_lstm(args):
    folder_path, base_output_dir, time_granularity = args
    output_dir = os.path.join(base_output_dir, os.path.basename(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc=f"Processing {folder_path}"):
        interpolate_data_with_lstm(csv_file, output_dir, time_granularity)
def process_folder(args):
    folder_path, base_output_dir, time_granularity = args
    output_dir = os.path.join(base_output_dir, os.path.basename(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc=f"Processing {folder_path}"):
        interpolate_data(csv_file, output_dir, time_granularity)

def main():
    # 处理以分钟为单位的 byDayLine(min)
    base_dir_min = 'byDayLine-(min)'
    base_output_dir_min = 'byDayLineConstant-lstm(min)'
    folders_min = [os.path.join(base_dir_min, d) for d in os.listdir(base_dir_min) if os.path.isdir(os.path.join(base_dir_min, d))]

    # 单线程逐个处理文件夹
    for folder in tqdm(folders_min, desc="Processing byDayLine(min)"):
        process_folder_lstm((folder, base_output_dir_min, 'min'))

if __name__ == '__main__':
    main()
