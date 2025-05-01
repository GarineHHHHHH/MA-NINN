import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取数据
def load_data(directory, start_date, end_date):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    df_list = []
    for file in all_files:
        date = os.path.basename(file).split('.')[0]
        if start_date <= date <= end_date:
            df = pd.read_csv(file)
            df['time'] = pd.to_datetime(date + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
            df.set_index('time', inplace=True)
            df_list.append(df)
    df = pd.concat(df_list)
    df.sort_index(inplace=True)  # 确保按时间顺序排序
    return df

# 创建序列数据
def create_sequences(data, feature_columns, target_columns, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[feature_columns].iloc[i:i+sequence_length].values)
        y.append(data[target_columns].iloc[i+sequence_length].values)
    return np.array(X), np.array(y)

def preprocess_data_with_lag_features(df, feature_columns, target_columns, sequence_length=60):
    # 计算衍生特征
    df['flow_intensity'] = df['dst_ip_count'].shift(1) * df['mean_packet_size'].shift(1)
    df['burst_ratio'] = df['max_packet_size'].shift(1) / (df['mean_packet_size'].shift(1) + 1e-6)
    df['rtt_stability'] = df['mean_rtt'].rolling(window=10).mean() / (df['std_rtt'].rolling(window=10).std() + 1e-6)

    # 填充缺失值
    df = df.ffill().bfill()

    # 更新特征列
    feature_columns += ['flow_intensity', 'burst_ratio', 'rtt_stability']

    # 归一化
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[feature_columns + target_columns])
    df_scaled = pd.DataFrame(df_scaled, columns=feature_columns + target_columns, index=df.index)

    # 创建序列数据
    X, y = create_sequences(df_scaled, feature_columns, target_columns, sequence_length)
    return X, y, scaler, feature_columns
# 数据预处理
def preprocess_data_without_lag_features(df, feature_columns, target_columns, sequence_length=60):
    df = df.ffill().bfill()  # 处理 NaN 值
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[feature_columns + target_columns])
    df_scaled = pd.DataFrame(df_scaled, columns=feature_columns + target_columns, index=df.index)

    X, y = create_sequences(df_scaled, feature_columns, target_columns, sequence_length)
    return X, y, scaler