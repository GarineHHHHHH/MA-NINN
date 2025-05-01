import os
import numpy as np
import pandas as pd
import subprocess
from lstmTools.loadData import load_data, preprocess_data_without_lag_features
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lstmTools.buildModel import build_LSTM_Attention_model, build_LSTM_Multi_Attention_model, build_LSTM_model, build_transformer_model, build_parallel_cnn_lstm_model, build_RNN_model, build_GRU_model, build_custom_pinn_lstm
from lstmTools.evaluateModel import evaluate_model, save_and_plot_training_history
from lstmTools.MultiAttention import build_custom_pinn_multi_lstm, build_custom_pinn_multi_lstm2, build_custom_pinn_multi_lstm3


def flatten_data(X):
    return X.reshape(X.shape[0], -1)

def log_gpu_memory(output_dir, phase):
    os.makedirs(output_dir, exist_ok=True)  # 确保目标目录存在
    log_file = os.path.join(output_dir, f'gpu_memory_{phase}.log')
    with open(log_file, 'w') as f:
        subprocess.run(['nvidia-smi'], stdout=f)

def main(method_used, batch_size, total_epochs, train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date, output_root_dir, attention_type, num_heads, key_dim, key_dim2):

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
        build_model = lambda input_shape, C: build_custom_pinn_multi_lstm(input_shape, C, lambda1=0.3, lambda2=0.1, lambda3=0.9, attention_type=attention_type, num_heads=num_heads, key_dim=key_dim)
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'pinn_lstm_multi2':
        build_model = lambda input_shape, C: build_custom_pinn_multi_lstm2(input_shape, C, lambda1=0.3, lambda2=0.1, lambda3=0.9, attention_type=attention_type, num_heads=num_heads, key_dim1=key_dim, key_dim2=key_dim2)
        preprocess_data = preprocess_data_without_lag_features
    elif method_used == 'pinn_lstm_multi3':
        build_model = lambda input_shape,output_shape : build_custom_pinn_multi_lstm3(input_shape, output_shape, attention_type=attention_type, num_heads=None, key_dim1=key_dim, key_dim2=key_dim2)
        preprocess_data = preprocess_data_without_lag_features

    

    directories = [os.path.join(input_root_dir, f'{direction}') for direction in ['A-C', 'C-A']]
    output_dirs = [os.path.join(output_root_dir, f'{direction}') for direction in ['A-C', 'C-A']]

    feature_columns = ['src_ip_count', 'dst_ip_count', 'mean_window_size', 'mean_packet_size', 'mean_session_duration']
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
            
            # 记录训练前的显存占用情况
            log_gpu_memory(output_dir, 'before_training')
            
            model = build_model(input_shape, C=57.6)  # 假设链路容量为1.0
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
            history = model.fit(X_train, y_train_combined, validation_data=(X_val, y_val_combined), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])

            # 记录训练后的显存占用情况
            log_gpu_memory(output_dir, 'after_training')
            
            # 调用新函数保存训练过程中的评估指标并绘制变化图
            save_and_plot_training_history(history, output_dir)

            evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)

        if method_used == 'pinn_lstm_multi3':
            for head1 in heads1:
                for head2 in heads2:
                    # 设置输出目录
                    output_dir = os.path.join(output_root_dir, f'heads1_{head1}_heads2_{head2}')
                    os.makedirs(output_dir, exist_ok=True)

                    # 记录训练前的显存占用情况
                    log_gpu_memory(output_dir, 'before_training')

                    # 构建模型
                    model = build_custom_pinn_multi_lstm3(
                        input_shape=input_shape,
                        output_shape=len(target_columns),
                        attention_type=attention_type,
                        num_heads1=head1,
                        num_heads2=head2,
                        key_dim1=key_dim,
                        key_dim2=key_dim2
                    )

                    # 定义回调函数
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

                    # 训练模型
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=total_epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, reduce_lr]
                    )

                    # 记录训练后的显存占用情况
                    log_gpu_memory(output_dir, 'after_training')

                    # 保存训练过程中的评估指标并绘制变化图
                    save_and_plot_training_history(history, output_dir)

                    # 评估模型
                    evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)
        else:
            # 记录训练前的显存占用情况
            log_gpu_memory(output_dir, 'before_training')
            
            model = build_model(input_shape, len(target_columns))

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])
            # history = model.fit(X_train, y_train_combined, validation_data=(X_val, y_val_combined), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])

            # 记录训练后的显存占用情况
            log_gpu_memory(output_dir, 'after_training')
            
            # 调用新函数保存训练过程中的评估指标并绘制变化图
            save_and_plot_training_history(history, output_dir)

            evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)



if __name__ == '__main__':
    method = 'pinn_lstm_multi3'
    batch = 128
    heads1 = [1, 2, 4, 8, 16]
    heads2 = [1, 2, 4, 8, 16]

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
        output_root_dir='../head-Results',
        attention_type='dot',
        num_heads=None,  # 不需要传递，因为 heads1 和 heads2 会覆盖
        key_dim=64,
        key_dim2=32
    )