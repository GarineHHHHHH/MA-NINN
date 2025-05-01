import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_gantt_chart(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将start_time和end_time字段转换为datetime类型
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])

        # 对src_ip进行编号
        df['src_ip_id'] = df['src_ip'].astype('category').cat.codes

        # 绘制甘特图
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(df['src_ip_id'].unique())))
        
        for idx, row in df.iterrows():
            color_idx = row['src_ip_id'] % len(colors)
            plt.plot([row['start_time'], row['end_time']], 
                     [row['src_ip_id'], row['src_ip_id']], 
                     marker='o', color=colors[color_idx], linewidth=1.5, alpha=0.7)

        # 设置日期格式
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # 设置标签和标题
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Source IP ID', fontsize=12)
        plt.title('Communication Time Periods for Each IP', fontsize=14)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.grid(axis='y', linestyle='-', alpha=0.5)
        plt.tight_layout()

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"Error plotting Gantt chart for {file_path}: {e}")

def plot_communication_frequency(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 对src_ip进行编号
        df['src_ip_id'] = df['src_ip'].astype('category').cat.codes

        # 统计每个IP的通信次数
        df['communication_count'] = df['dst_ip_count']

        # 绘制通信频率图
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('Blues')
        colors = cmap(np.linspace(0.3, 1, len(df['src_ip_id'].unique())))
        
        bars = plt.bar(df['src_ip_id'], df['communication_count'], color=colors, edgecolor='black', alpha=0.8)

        # 设置标签和标题
        plt.xlabel('Source IP ID', fontsize=12)
        plt.ylabel('Communication Frequency', fontsize=12)
        plt.title('Communication Frequency for Each IP', fontsize=14)
        plt.xticks(df['src_ip_id'], rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

        # 添加数值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8, color='black')

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"Error plotting communication frequency chart for {file_path}: {e}")

plot_gantt_chart('byIpSta/A-C/2024-12-28.csv')