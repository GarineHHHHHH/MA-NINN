import os
import pandas as pd

# 定义天数和目录
days = [2, 4, 6, 8, 10]
base_dir = '/home/zhaohuan/桌面/outputPredict{day}Days'

# 初始化数据字典
data_ac = {'days': days}
data_ca = {'days': days}

# 遍历每个天数目录
for day in days:
    file_path = os.path.join(base_dir.format(day=day), 'sorted_metrics_summary.csv')
    df = pd.read_csv(file_path)
    
    # 提取A-C方向的数据
    df_ac = df[df['Direction'] == 'A-C']
    for method in df_ac['Method']:
        if f'RMSE-{method}' not in data_ac:
            data_ac[f'RMSE-{method}'] = []
            data_ac[f'MAE-{method}'] = []
            data_ac[f'MAPE-{method}'] = []
        data_ac[f'RMSE-{method}'].append(df_ac[df_ac['Method'] == method]['RMSE'].values[0])
        data_ac[f'MAE-{method}'].append(df_ac[df_ac['Method'] == method]['MAE'].values[0])
        data_ac[f'MAPE-{method}'].append(df_ac[df_ac['Method'] == method]['MAPE'].values[0])
    
    # 提取C-A方向的数据
    df_ca = df[df['Direction'] == 'C-A']
    for method in df_ca['Method']:
        if f'RMSE-{method}' not in data_ca:
            data_ca[f'RMSE-{method}'] = []
            data_ca[f'MAE-{method}'] = []
            data_ca[f'MAPE-{method}'] = []
        data_ca[f'RMSE-{method}'].append(df_ca[df_ca['Method'] == method]['RMSE'].values[0])
        data_ca[f'MAE-{method}'].append(df_ca[df_ca['Method'] == method]['MAE'].values[0])
        data_ca[f'MAPE-{method}'].append(df_ca[df_ca['Method'] == method]['MAPE'].values[0])

# 转换为DataFrame
df_ac = pd.DataFrame(data_ac)
df_ca = pd.DataFrame(data_ca)

# 保存为CSV文件
df_ac.to_csv('/home/zhaohuan/桌面/A-C-DiffDaysPredict.csv', index=False)
df_ca.to_csv('/home/zhaohuan/桌面/C-A-DiffDaysPredict.csv', index=False)

print("CSV files have been created successfully.")