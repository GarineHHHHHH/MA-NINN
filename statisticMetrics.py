import os
import pandas as pd

def extract_metrics(input_root_dir, methods, directions, output_file):
    results = []

    for method in methods:
        for direction in directions:
            metrics_file = os.path.join(input_root_dir, method, direction, 'metrics.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                rmse = df.loc[df['metric'] == 'mean_rtt', 'RMSE'].values[0]
                mae = df.loc[df['metric'] == 'mean_rtt', 'MAE'].values[0]
                mape = df.loc[df['metric'] == 'mean_rtt', 'MAPE'].values[0]
                smape = df.loc[df['metric'] == 'mean_rtt', 'SMAPE'].values[0]
                r2 = df.loc[df['metric'] == 'mean_rtt', 'R2'].values[0]
                results.append([method, direction, rmse, mae, mape, smape, r2])
            else:
                print(f"Metrics file not found for method: {method}, direction: {direction}")

    results_df = pd.DataFrame(results, columns=['Method', 'Direction', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'R2'])
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results_df.to_csv(output_file, index=False)


def sort_and_save_metrics(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    
    # 按照 Direction 分类
    df_ac = df[df['Direction'] == 'A-C']
    df_ca = df[df['Direction'] == 'C-A']
    
    # 按照 Method 排序
    method_order = ['rnn', 'gru', 'transformer', 'lstm', 'cnn_lstm', 'attention', 'multi', 'pinn_lstm', 'pinn_lstm_multi']
    df_ac['Method'] = pd.Categorical(df_ac['Method'], categories=method_order, ordered=True)
    df_ca['Method'] = pd.Categorical(df_ca['Method'], categories=method_order, ordered=True)
    
    df_ac = df_ac.sort_values('Method')
    df_ca = df_ca.sort_values('Method')
    
    # 合并分类后的数据
    sorted_df = pd.concat([df_ac, df_ca])
    
    # 保存到新的 CSV 文件
    sorted_df.to_csv(output_file, index=False)

def different_days_prediction_metrics_summary():
    days=[2,4,6,8,10]
    for day in days:
        input_root_dir = f'../outputPredict{day}Days'
        methods = ['attention', 'cnn_lstm', 'gru', 'lstm', 'multi', 'pinn_lstm', 'pinn_lstm_multi', 'rnn', 'transformer']
        directions = ['A-C', 'C-A']
        output_file = f'../outputPredict{day}Days/metrics_summary.csv'
        
        extract_metrics(input_root_dir, methods, directions, output_file)
        input_file = f'../outputPredict{day}Days/metrics_summary.csv'
        output_file = f'../outputPredict{day}Days/sorted_metrics_summary.csv'
        sort_and_save_metrics(input_file, output_file)

def aggregate_and_sort_metrics(root_dirs, output_files):
    columns = ['direction', 'headnum', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'R2']

    for root_dir, output_file in zip(root_dirs, output_files):
        data = []

        for headnum in range(2, 11):
            for direction in ['A-C', 'C-A']:
                metrics_file = os.path.join(root_dir, f'{headnum}-heads', direction, 'metrics.csv')
                if os.path.exists(metrics_file):
                    df = pd.read_csv(metrics_file)
                    rmse = df.loc[df['metric'] == 'mean_rtt', 'RMSE'].values[0]
                    mae = df.loc[df['metric'] == 'mean_rtt', 'MAE'].values[0]
                    mape = df.loc[df['metric'] == 'mean_rtt', 'MAPE'].values[0]
                    smape = df.loc[df['metric'] == 'mean_rtt', 'SMAPE'].values[0]
                    r2 = df.loc[df['metric'] == 'mean_rtt', 'R2'].values[0]
                    data.append([direction, headnum, rmse, mae, mape, smape, r2])

        df_output = pd.DataFrame(data, columns=columns)
        df_output.to_csv(output_file, index=False)
import pandas as pd

def select_best_headnum(file_path):
    df = pd.read_csv(file_path)
    
    # 计算综合评分
    df['score'] = df['RMSE'] + df['MAE'] + df['MAPE'] - df['R2']
    
    # 按照综合评分排序，选择评分最低的 headnum
    best_row = df.loc[df['score'].idxmin()]
    best_headnum = best_row['headnum']
    
    return best_headnum, best_row
import os
import pandas as pd

def extract_metrics_by_direction(days, root_dir, output_ac_file, output_ca_file):
    # 初始化存储数据的列表
    ac_data = []
    ca_data = []

    for day in days:
        # 构造 A-C 和 C-A 的 metrics 文件路径
        ac_metrics_file = os.path.join(root_dir, f"{day}Days", "A-C", "metrics.csv")
        ca_metrics_file = os.path.join(root_dir, f"{day}Days", "C-A", "metrics.csv")

        # 提取 A-C 的 metrics 数据
        if os.path.exists(ac_metrics_file):
            df_ac = pd.read_csv(ac_metrics_file)
            row = df_ac.loc[df_ac['metric'] == 'mean_rtt'].iloc[0]  # 提取 mean_rtt 行
            ac_data.append([day, row['MSE'], row['RMSE'], row['MAE'], row['MAPE'], row['SMAPE'], row['R'], row['R2']])

        # 提取 C-A 的 metrics 数据
        if os.path.exists(ca_metrics_file):
            df_ca = pd.read_csv(ca_metrics_file)
            row = df_ca.loc[df_ca['metric'] == 'mean_rtt'].iloc[0]  # 提取 mean_rtt 行
            ca_data.append([day, row['MSE'], row['RMSE'], row['MAE'], row['MAPE'], row['SMAPE'], row['R'], row['R2']])

    # 将数据转换为 DataFrame 并保存为 CSV 文件
    ac_df = pd.DataFrame(ac_data, columns=['day', 'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'R', 'R2'])
    ca_df = pd.DataFrame(ca_data, columns=['day', 'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'R', 'R2'])

    # 保存到指定的输出文件
    ac_df.to_csv(output_ac_file, index=False)
    ca_df.to_csv(output_ca_file, index=False)

import os
import pandas as pd

def extract_all_metrics(days, methods, root_dir, output_ac_file, output_ca_file):
    # 初始化存储数据的字典
    ac_data = {'day': []}
    ca_data = {'day': []}

    for method in methods:
        # 为每种方法添加列
        ac_data[f'RMSE_{method}'] = []
        ac_data[f'MAE_{method}'] = []
        ac_data[f'MAPE_{method}'] = []
        ca_data[f'RMSE_{method}'] = []
        ca_data[f'MAE_{method}'] = []
        ca_data[f'MAPE_{method}'] = []

    for day in days:
        ac_data['day'].append(day)
        ca_data['day'].append(day)

        for method in methods:
            # 构造 A-C 和 C-A 的 metrics 文件路径
            ac_metrics_file = os.path.join(root_dir, f"{day}Days", method, "A-C", "metrics.csv")
            ca_metrics_file = os.path.join(root_dir, f"{day}Days", method, "C-A", "metrics.csv")

            # 提取 A-C 的 metrics 数据
            if os.path.exists(ac_metrics_file):
                df_ac = pd.read_csv(ac_metrics_file)
                row = df_ac.loc[df_ac['metric'] == 'mean_rtt'].iloc[0]  # 提取 mean_rtt 行
                ac_data[f'RMSE_{method}'].append(row['RMSE'])
                ac_data[f'MAE_{method}'].append(row['MAE'])
                ac_data[f'MAPE_{method}'].append(row['MAPE'])
            else:
                # 如果文件不存在，填充 NaN
                ac_data[f'RMSE_{method}'].append(None)
                ac_data[f'MAE_{method}'].append(None)
                ac_data[f'MAPE_{method}'].append(None)

            # 提取 C-A 的 metrics 数据
            if os.path.exists(ca_metrics_file):
                df_ca = pd.read_csv(ca_metrics_file)
                row = df_ca.loc[df_ca['metric'] == 'mean_rtt'].iloc[0]  # 提取 mean_rtt 行
                ca_data[f'RMSE_{method}'].append(row['RMSE'])
                ca_data[f'MAE_{method}'].append(row['MAE'])
                ca_data[f'MAPE_{method}'].append(row['MAPE'])
            else:
                # 如果文件不存在，填充 NaN
                ca_data[f'RMSE_{method}'].append(None)
                ca_data[f'MAE_{method}'].append(None)
                ca_data[f'MAPE_{method}'].append(None)

    # 将数据转换为 DataFrame 并保存为 CSV 文件
    ac_df = pd.DataFrame(ac_data)
    ca_df = pd.DataFrame(ca_data)

    ac_df.to_csv(output_ac_file, index=False)
    ca_df.to_csv(output_ca_file, index=False)

if __name__ == '__main__':
    # 定义天数和方法
    days = [2, 4, 6, 8, 10]
    methods = ['rnn', 'gru', 'transformer', 'lstm', 'cnn_lstm', 'attention', 'multi', 'pinn_lstm', 'pinn_lstm_multi']
    root_dir = '/home/zhaohuan/桌面'

    # 输出文件路径
    output_ac_file = os.path.join(root_dir, "A-C.csv")
    output_ca_file = os.path.join(root_dir, "C-A.csv")

    # 提取并保存 metrics 数据
    extract_all_metrics(days, methods, root_dir, output_ac_file, output_ca_file)