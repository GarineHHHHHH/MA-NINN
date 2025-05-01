import os
import numpy as np
import pandas as pd
import subprocess
import psutil
from lstmTools.loadData import load_data, preprocess_data_without_lag_features
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lstmTools.buildModel import build_LSTM_Attention_model, build_LSTM_Multi_Attention_model, build_LSTM_model, build_transformer_model, build_parallel_cnn_lstm_model, build_RNN_model, build_GRU_model, build_custom_pinn_lstm
from lstmTools.evaluateModel import evaluate_model, save_and_plot_training_history
from lstmTools.MultiAttention import build_custom_pinn_multi_lstm, build_custom_pinn_multi_lstm2, build_custom_pinn_multi_lstm3, build_custom_pinn_multi_lstm3_layer_contral
import time
def flatten_data(X):
    return X.reshape(X.shape[0], -1)

def log_gpu_memory(output_dir, phase):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'gpu_memory_{phase}.log')
    with open(log_file, 'w') as f:
        subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv'], stdout=f)

def log_cpu_usage(output_dir, phase):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'cpu_usage_{phase}.log')
    with open(log_file, 'w') as f:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        f.write(f"CPU Usage: {cpu_percent}%\n")
        f.write(f"Memory Usage: {memory_info.percent}%\n")

def log_experiment_summary(output_root_dir, head1, head2, elapsed_time):
    summary_file = os.path.join(output_root_dir, 'experiment_summary.log')
    with open(summary_file, 'a') as f:
        f.write(f"Head1: {head1}, Head2: {head2}\n")
        f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
        f.write("-" * 50 + "\n")
def main(method_used, batch_size, total_epochs, train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date, output_root_dir, attention_type, num_head1,num_head2, key_dim1, key_dim2):

    input_root_dir = 'byDayLineConstant-(min)'

    # build_model = lambda input_shape,output_shape : build_custom_pinn_multi_lstm3(input_shape, output_shape, attention_type, num_head1,num_head2, key_dim1, key_dim2)
    build_model = lambda input_shape,output_shape : build_custom_pinn_multi_lstm3_layer_contral(input_shape, output_shape, attention_type, num_head1,num_head2, key_dim1, key_dim2)
    preprocess_data = preprocess_data_without_lag_features

    

    directories = [os.path.join(input_root_dir, f'{direction}') for direction in ['A-C', 'C-A']]
    output_dirs = [os.path.join(output_root_dir, f'{direction}') for direction in ['A-C', 'C-A']]

    feature_columns = ['src_ip_count', 'dst_ip_count', 'mean_window_size', 'mean_packet_size', 'mean_session_duration']
    target_columns = ['mean_rtt']

    for directory, output_dir in zip(directories, output_dirs):
        start_time = time.time()
        df_train = load_data(directory, train_start_date, train_end_date)
        df_val = load_data(directory, val_start_date, val_end_date)
        df_test = load_data(directory, test_start_date, test_end_date)

        X_train, y_train, scaler = preprocess_data(df_train, feature_columns, target_columns)
        X_val, y_val, _ = preprocess_data(df_val, feature_columns, target_columns)
        X_test, y_test, _ = preprocess_data(df_test, feature_columns, target_columns)
        updated_feature_columns = feature_columns
        input_shape = (X_train.shape[1], X_train.shape[2])
    
        # 记录资源使用（训练前）
        log_gpu_memory(output_dir, 'before_training')
        log_cpu_usage(output_dir, 'before_training')

        model = build_model(input_shape, len(target_columns))

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])
        # history = model.fit(X_train, y_train_combined, validation_data=(X_val, y_val_combined), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])

        # 记录资源使用（训练后）
        log_gpu_memory(output_dir, 'after_training')
        log_cpu_usage(output_dir, 'after_training')
        # 记录运行时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(os.path.join(output_dir, 'runtime.log'), 'w') as f:
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")

        # 汇总实验结果
        log_experiment_summary(output_root_dir, num_head1, num_head2, elapsed_time)
        # 调用新函数保存训练过程中的评估指标并绘制变化图
        save_and_plot_training_history(history, output_dir)

        evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)

if __name__ == '__main__':
    method = 'pinn_lstm_multi3'
    batch = 16
    heads1 = 8
    heads2 = 2
    keydim_combinations = [
        (16, 8),
        (32, 16),
        (32, 8),
        (64, 32),
        (64, 16),
        (64, 8)
    ]

    for key_dim1, key_dim2 in keydim_combinations:

        main(
            method_used=method,
            batch_size=batch,
            total_epochs=30,
            train_start_date='2025-01-01',
            train_end_date='2025-01-05',
            val_start_date='2025-01-06',
            val_end_date='2025-01-06',
            test_start_date='2025-01-07',
            test_end_date='2025-01-07',
            output_root_dir=f'../keydim-SimpleLab/keydim1_{key_dim1}_keydim2_{key_dim2}',
            attention_type='dot',
            num_head1=heads1,
            num_head2=heads2,
            key_dim1=key_dim1,
            key_dim2=key_dim2
        )