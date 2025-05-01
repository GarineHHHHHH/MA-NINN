from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, MultiHeadAttention, RepeatVector, Input
from tensorflow.keras.layers import LayerNormalization, Add
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SimpleRNN, GRU, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def custom_loss(C, lambda1=0.1, lambda2=0.1, lambda3=0.1):
    def loss(y_true, y_pred):
        # 数据损失
        mse_loss = tf.reduce_mean(tf.square(y_true[:, -1] - y_pred))
        
        # 提取特定特征
        src_ip = y_true[:, 0]  # 假设 src_ip_count 是第一个特征
        mean_pkt = y_true[:, 3]  # 假设 mean_packet_size 是第二个特征

        # 时延-负载损失
        flow_intensity = src_ip * mean_pkt
        flow_intensity_mean = tf.reduce_mean(flow_intensity, keepdims=True)  # 计算每个样本的平均流量强度
        rtt_load_loss = tf.reduce_mean(tf.square(y_pred - 0.5 * flow_intensity_mean / C))
        
        # 突发衰减损失（自动微分计算时间导数）
        time = tf.range(tf.shape(y_pred)[0], dtype=tf.float32)  # 确保 time 的形状和类型正确
        time = tf.reshape(time, (-1, 1))
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(time)
            max_rtt = y_pred  # 确保形状兼容
        d_max_rtt = t.gradient(max_rtt, time)
        
        # 如果 d_max_rtt 为 None，则用 0 替换
        if d_max_rtt is None:
            d_max_rtt = tf.zeros_like(max_rtt)
        
        decay_loss = tf.reduce_mean(tf.square(d_max_rtt + 0.1 * (max_rtt - y_pred)))

        # RTT-窗口损失
        mean_win = y_true[:, 2]
        window_loss = tf.reduce_mean(tf.square(y_pred * mean_win/100.0))
        
        # 总损失
        total_loss = mse_loss + lambda1 * decay_loss + lambda3 * window_loss + lambda2 * rtt_load_loss
        return total_loss
    return loss

def build_custom_pinn_lstm(input_shape, C, lambda1, lambda2, lambda3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(1))  # 输出 mean_rtt
    model.compile(optimizer='adam', loss=custom_loss(C, lambda1, lambda2, lambda3))
    return model

def build_custom_pinn_multi_lstm(input_shape, C, lambda1, lambda2, lambda3):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=True)(x)
    attn_output = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = LSTM(32, return_sequences=False)(attn_output)
    outputs = Dense(1)(x)  # 输出 mean_rtt
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=custom_loss(C, lambda1, lambda2, lambda3))
    return model
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
    
    # 将预测结果和真实值拼接到零矩阵上
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

    # 返回验证集上的损失作为评估指标
    return metrics_df['MSE'].mean()