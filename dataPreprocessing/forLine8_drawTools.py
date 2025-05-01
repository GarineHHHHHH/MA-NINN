import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_gantt_chart(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将所有时间字段转换为datetime类型
        for col in df.columns:
            if col.endswith('_start_time') or col.endswith('_end_time'):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # 过滤掉没有任何时间段的IP
        df = df.dropna(subset=[col for col in df.columns if col.endswith('_start_time')])

        # 绘制甘特图
        fig, ax = plt.subplots(figsize=(15, 10))

        colors = plt.cm.get_cmap('tab20', len(df['src_ip'].unique()))

        for idx, row in df.iterrows():
            color = colors(idx % len(df['src_ip'].unique()))
            for i in range(1, 101):  # 假设最多有100个时间段
                start_col = f'D{i}_start_time'
                end_col = f'D{i}_end_time'
                if pd.notna(row[start_col]) and pd.notna(row[end_col]):
                    ax.plot([row[start_col], row[end_col]], [row['src_ip'], row['src_ip']], marker='o', color=color)

        # 设置日期格式
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # 设置标签和标题
        plt.xlabel('Time')
        plt.ylabel('Source IP')
        plt.title('Communication Time Periods for Each IP')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"Error plotting Gantt chart for {file_path}: {e}")

def plot_communication_metrics(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 绘制各个IP的通信指标对比图
        fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

        metrics = ['D_mean_window_size', 'D_mean_packet_size', 'D_mean_rtt', 'D_mean_session_duration']
        titles = ['Mean Window Size (B)', 'Mean Packet Size (B)', 'Mean RTT (ms)', 'Mean Session Duration (s)']

        for ax, metric, title in zip(axes, metrics, titles):
            ax.bar(df['src_ip'], df[metric])
            ax.set_ylabel(title)
            ax.grid(True)

        plt.xlabel('Source IP')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 显示图像
        plt.show()

    except Exception as e:
        print(f"Error plotting communication metrics for {file_path}: {e}")

# 示例调用
if __name__ == '__main__':
    example_file = '2024-12-29.csv'
    plot_gantt_chart(example_file)
    plot_communication_metrics(example_file)