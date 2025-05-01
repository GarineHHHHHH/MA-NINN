import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# 计算 SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

# 保存训练过程中的评估指标并绘制变化图
def save_and_plot_training_history(history, output_dir):
    # 保存训练过程中的评估指标
    history_df = pd.DataFrame(history.history)
    os.makedirs(output_dir, exist_ok=True)
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # 绘制训练过程中的评估指标变化
    plt.figure(figsize=(16, 9))
    plt.plot(history.history['loss'], label='Training Loss', color='tab:blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange')
    plt.title('Training and Validation Loss', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Epoch', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Loss', fontsize=12, fontname='Times New Roman')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    os.makedirs(os.path.join(output_dir, 'pics'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pics', 'training_validation_loss.png'))
    plt.close()
def evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, feature_columns):
    y_pred = model.predict(X_test)
    
    # 创建一个与原始数据形状匹配的零矩阵
    zeros = np.zeros((y_test.shape[0], len(feature_columns)))
    
    # 将预测值和真实值拼接到零矩阵上
    y_test_combined = np.concatenate((zeros, y_test), axis=1)
    y_pred_combined = np.concatenate((zeros, y_pred), axis=1)
    

    # 进行逆变换
    y_test_inv = scaler.inverse_transform(y_test_combined)
    y_pred_inv = scaler.inverse_transform(y_pred_combined)

    # 只保留目标列
    y_test_inv = y_test_inv[:, -len(target_columns):]
    y_pred_inv = y_pred_inv[:, -len(target_columns):]

    # 保存预测结果
    result_df = pd.DataFrame(y_pred_inv, columns=[f'pred_{col}' for col in target_columns])
    result_df['time'] = pd.date_range(start='2025-01-11', periods=len(result_df), freq='min')
    result_df.set_index('time', inplace=True)
    result_df.to_csv(os.path.join(output_dir, 'predictions.csv'))

    # 计算评估指标
    metrics = []
    for i, col in enumerate(target_columns):
        mse = mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
        mape = mean_absolute_percentage_error(y_test_inv[:, i], y_pred_inv[:, i])
        smape = symmetric_mean_absolute_percentage_error(y_test_inv[:, i], y_pred_inv[:, i])
        r2 = r2_score(y_test_inv[:, i], y_pred_inv[:, i])
        r = np.corrcoef(y_test_inv[:, i], y_pred_inv[:, i])[0, 1]
        metrics.append({'metric': col, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'SMAPE': smape, 'R': r, 'R2': r2})
        print(f'{col} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, SMAPE: {smape}, R: {r}, R2: {r2}')

        plt.figure(figsize=(16, 9))
        plt.plot(y_test_inv[:, i], label='True', color='tab:blue')
        plt.plot(y_pred_inv[:, i], label='Predicted', color='tab:orange')
        plt.title(f'{col} Prediction', fontsize=14, fontname='DejaVu Sans')
        plt.xlabel('Time', fontsize=12, fontname='DejaVu Sans')
        plt.ylabel(col, fontsize=12, fontname='DejaVu Sans')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.minorticks_on()
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        os.makedirs(os.path.join(output_dir, 'pics'), exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'pics', f'{col}.png'))
        plt.close()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    