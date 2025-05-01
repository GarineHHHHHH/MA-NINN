import os
import numpy as np
import pandas as pd
from lstmTools.loadData import load_data, preprocess_data_without_lag_features
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lstmTools.superParameter import evaluate_model, save_and_plot_training_history
from lstmTools.MultiAttention import build_custom_pinn_multi_lstm3


def flatten_data(X):
    return X.reshape(X.shape[0], -1)

def train_and_evaluate(method_used, batch_size, total_epochs, train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date, output_root_dir, C, lambda1, lambda2, lambda3):

    input_root_dir = 'byDayLineConstant-(min)'

    preprocess_data = preprocess_data_without_lag_features

    directories = [os.path.join(input_root_dir, f'{direction}') for direction in ['A-C', 'C-A']]
    output_dirs = [os.path.join(output_root_dir, f'l1{lambda1}-l2{lambda2}-l3{lambda3}', f'{direction}') for direction in ['A-C', 'C-A']]

    feature_columns = ['src_ip_count', 'dst_ip_count', 'mean_window_size', 'mean_packet_size', 'mean_session_duration']
    target_columns = ['mean_rtt']

    scores = []

    for directory, output_dir in zip(directories, output_dirs):
        df_train = load_data(directory, train_start_date, train_end_date)
        df_val = load_data(directory, val_start_date, val_end_date)
        df_test = load_data(directory, test_start_date, test_end_date)

        X_train, y_train, scaler = preprocess_data(df_train, feature_columns, target_columns)
        X_val, y_val, _ = preprocess_data(df_val, feature_columns, target_columns)
        X_test, y_test, _ = preprocess_data(df_test, feature_columns, target_columns)
        updated_feature_columns = feature_columns
        input_shape = (X_train.shape[1], X_train.shape[2])

        if directory.endswith('A-C'):
            attention_type = 'add'
            num_heads = 4
        else:
            attention_type = 'dot'
            num_heads = 6

        if method_used == 'pinn_lstm' or method_used == 'pinn_lstm_multi' or method_used == 'pinn_lstm_multi2':
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
            model = build_custom_pinn_multi_lstm2(input_shape, C, lambda1, lambda2, lambda3, attention_type=attention_type, num_heads=num_heads, key_dim1=64, key_dim2=32)
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
            history = model.fit(X_train, y_train_combined, validation_data=(X_val, y_val_combined), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])
            # 调用新函数保存训练过程中的评估指标并绘制变化图
            save_and_plot_training_history(history, output_dir)

            score = evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)
            scores.append(score)

        else:
            model = build_custom_pinn_multi_lstm3(input_shape, C, lambda1, lambda2, lambda3, attention_type=attention_type, num_heads=num_heads, key_dim1=64, key_dim2=32)

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=total_epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr])
            # 调用新函数保存训练过程中的评估指标并绘制变化图
            save_and_plot_training_history(history, output_dir)

            score = evaluate_model(model, X_test, y_test, scaler, target_columns, output_dir, updated_feature_columns)
            scores.append(score)

    return scores
def grid_search(method_used, batch_size, total_epochs, train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date, output_root_dir, C, lambda1_values, lambda2_values, lambda3_values):
    best_score = float('inf')
    best_params = None
    results = []

    # 确保输出目录存在
    os.makedirs(output_root_dir, exist_ok=True)
    results_file = os.path.join(output_root_dir, 'grid_search_results.csv')

    # 如果文件不存在，写入表头
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write('lambda1,lambda2,lambda3,mean_score,std_score,max_score,min_score\n')

    for lambda1 in lambda1_values:
        for lambda2 in lambda2_values:
            for lambda3 in lambda3_values:
                print(f'Training with lambda1={lambda1}, lambda2={lambda2}, lambda3={lambda3}')
                all_scores = []
                for _ in range(5):  # 进行10次重复实验
                    scores = train_and_evaluate(method_used, batch_size, total_epochs, train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date, output_root_dir, C, lambda1, lambda2, lambda3)
                    all_scores.extend(scores)
                
                # 计算统计指标
                mean_score = np.mean(all_scores)
                std_score = np.std(all_scores)
                max_score = np.max(all_scores)
                min_score = np.min(all_scores)
                
                # 将结果添加到内存中的列表
                results.append({'lambda1': lambda1, 'lambda2': lambda2, 'lambda3': lambda3, 'mean_score': mean_score, 'std_score': std_score, 'max_score': max_score, 'min_score': min_score})
                
                # 更新最佳参数
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = (lambda1, lambda2, lambda3)

                # 将当前结果写入文件
                with open(results_file, 'a') as f:
                    f.write(f'{lambda1},{lambda2},{lambda3},{mean_score},{std_score},{max_score},{min_score}\n')

    print(f'Best score: {best_score} with parameters: lambda1={best_params[0]}, lambda2={best_params[1]}, lambda3={best_params[2]}')

if __name__ == '__main__':
    methods = ['pinn_lstm_multi3']
    batch = 128
    C = 57.6
    lambda1_values = [0.4, 0.6, 0.8]
    lambda2_values = [0.4, 0.6, 0.8]
    lambda3_values = [ 0.4, 0.6, 0.8]

    for method in methods:
        grid_search(method, batch_size=batch, total_epochs=10,
                    train_start_date='2025-01-01', train_end_date='2025-01-10',
                    val_start_date='2025-01-11', val_end_date='2025-01-12',
                    test_start_date='2025-01-13', test_end_date='2025-01-14',
                    output_root_dir='../outputLongSuperParameter3', C=C, lambda1_values=lambda1_values, lambda2_values=lambda2_values, lambda3_values=lambda3_values)