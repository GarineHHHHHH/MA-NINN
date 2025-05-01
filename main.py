import os
import numpy as np
import pandas as pd
from lstmTools.loadData import load_data, preprocess_data_without_lag_features
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lstmTools.buildModel import build_LSTM_Attention_model, build_LSTM_Multi_Attention_model, build_LSTM_model, build_transformer_model, build_parallel_cnn_lstm_model, build_custom_pinn_multi_lstm, build_custom_pinn_lstm, build_RNN_model, build_GRU_model
from lstmTools.evaluateModel import evaluate_model, save_and_plot_training_history


def flatten_data(X):
    return X.reshape(X.shape[0], -1)

def main(method_used, batch_size, total_epochs, train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date, output_root_dir):

    input_root_dir = 'byDayLineConstant-(min)'

    if method_used == 'lstm':
        build_model = build_LSTM_model
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'attention':
        build_model = build_LSTM_Attention_model
        preprocess_data = preprocess_data_without_lag_features  
    elif method_used == 'multi':
        build_model = build_LSTM_Multi_Attention_model
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'rnn':
        build_model = build_RNN_model
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'gru':
        build_model = build_GRU_model
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'transformer':
        build_model = build_transformer_model
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'cnn_lstm':
        build_model = build_parallel_cnn_lstm_model
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'pinn_lstm':
        build_model = build_custom_pinn_lstm
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'pinn_lstm_multi':
        build_model = build_custom_pinn_multi_lstm
        preprocess_data = preprocess_data_without_lag_features

    directories = [os.path.join(input_root_dir, f'{direation}') for direation in ['A-C', 'C-A']]
    output_dirs = [os.path.join(output_root_dir, f'{method_used}', f'{direation}') for direation in ['A-C', 'C-A']]

    feature_columns = ['src_ip_count', 'dst_ip_count', 'mean_window_size', 'mean_packet_size',  'mean_session_duration']
    target_columns = ['mean_rtt']

    for directory, output_dir in zip(directories, output_dirs):
        df_train = load_data(directory, train_start_date, train_end_date)
        df_val = load_data(directory, val_start_date, val_end_date)
        df_test = load_data(directory, test_start_date, test_end_date)

        X_train, y_train, scaler = preprocess_data(df_train, feature_columns, target_columns)
        X_val, y_val, _ = preprocess_data(df_val, feature_columns, target_columns)
        X_test, y_test, _ = preprocess_data(df_test, feature_columns, target_columns)
        updated_feature_columns = feature_columns
        input_shape = (X_train.shape[1], X_train.shape[2])

        if method_used == 'pinn_lstm' or method_used == 'pinn_lstm_multi':
            # 扩展 y_train, y_val, y_test 的维度
            y_train_expanded = np.expand_dims(y_train, axis=-1)
            y_val_expanded = np.expand_dims(y_val, axis=-1)
            y_test_expanded = np.expand_dims(y_test, axis=-1)
            
            # 扩展 y_train_expanded 的维度以匹配 X_train
            y_train_expanded = np.tile(y_train_expanded, (1, X_train.shape[1], 1))
            y_val_expanded = np.tile(y_val_expanded, (1, X_val.shape[1], 1))
            y_test_expanded = np.tile(y_test_expanded, (1, X_test.shape[1], 1))
            
            # 合并特征列和目标列
            y_train_combined = np.concatenate([X_train, y_train_expanded], axis=-1)
            y_val_combined = np.concatenate([X_val, y_val_expanded], axis=-1)
            y_test_combined = np.concatenate([X_test, y_test_expanded], axis=-1)
            model = build_model(input_shape, C=1.0)  # 假设链路容量为1.0
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
            history = model.fit(X_train, y_train_combined, validation_data=(X_val, y_val_combined), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])
            # 调用新函数保存训练过程中的评估指标并绘制变化图
            save_and_plot_training_history(history, output_dir)

            evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)

        else:
            model = build_model(input_shape, len(target_columns))

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])
            # 调用新函数保存训练过程中的评估指标并绘制变化图
            save_and_plot_training_history(history, output_dir)

            evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)

if __name__ == '__main__':
    methods = ['pinn_lstm_multi', 'pinn_lstm', 'lstm', 'attention', 'multi', 'rnn', 'gru', 'transformer', 'cnn_lstm']
    batch = 16


    for method in methods:
        main(method, batch_size=batch, total_epochs=30,
                train_start_date='2025-01-01', train_end_date='2025-01-05',
                val_start_date='2025-01-06', val_end_date='2025-01-06',
                    test_start_date='2025-01-07', test_end_date='2025-01-08',
                    output_root_dir=f'../outputPredict2Days')
    for method in methods:
        main(method, batch_size=batch, total_epochs=30,
                train_start_date='2025-01-01', train_end_date='2025-01-05',
                val_start_date='2025-01-06', val_end_date='2025-01-06',
                    test_start_date='2025-01-07', test_end_date='2025-01-10',
                    output_root_dir=f'../outputPredict4Days')
    for method in methods:
        main(method, batch_size=batch, total_epochs=30,
                train_start_date='2025-01-01', train_end_date='2025-01-05',
                val_start_date='2025-01-06', val_end_date='2025-01-06',
                    test_start_date='2025-01-07', test_end_date='2025-01-12',
                    output_root_dir=f'../outputPredict6Days')
    for method in methods:
        main(method, batch_size=batch, total_epochs=30,
                 train_start_date='2025-01-01', train_end_date='2025-01-05',
                val_start_date='2025-01-06', val_end_date='2025-01-06',
                    test_start_date='2025-01-07', test_end_date='2025-01-14',
                    output_root_dir=f'../outputPredict8Days')
    for method in methods:
        main(method, batch_size=batch, total_epochs=30,
                train_start_date='2025-01-01', train_end_date='2025-01-05',
                val_start_date='2025-01-06', val_end_date='2025-01-06',
                    test_start_date='2025-01-07', test_end_date='2025-01-16',
                    output_root_dir=f'../outputPredict10Days')