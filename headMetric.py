import os
import csv

def parse_gpu_log(file_path):
    """
    解析 GPU 日志文件，提取 utilization.gpu 和 memory.used 的值
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None, None  # 如果文件内容不足，返回 None
        gpu_utilization = int(lines[1].split(',')[0].strip().replace('%', ''))
        memory_used = int(lines[1].split(',')[1].strip().replace('MiB', ''))
        return gpu_utilization, memory_used

def parse_runtime_log(file_path):
    """
    解析 runtime.log 文件，提取 Elapsed time 的值
    """
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        if "Elapsed time:" in line:
            elapsed_time = float(line.split(":")[1].strip().replace("seconds", "").strip())
            return elapsed_time
    return None

def parse_metrics_file(file_path):
    """
    解析 metrics.csv 文件，提取 RMSE, MAE, MAPE 的值
    """
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['metric'] == 'mean_rtt':  # 只处理 mean_rtt 行
                rmse = float(row['RMSE'])
                mae = float(row['MAE'])
                mape = float(row['MAPE'])
                return rmse, mae, mape
    return None, None, None

def process_directory_by_subdir(root_dir, sub_dir, include_runtime=False, include_metrics=False):
    """
    遍历目录结构，统计指定子目录 (A-C 或 C-A) 的 GPU 和内存变化，以及运行时间和指标
    """
    results = []
    for head_dir in os.listdir(root_dir):
        head_path = os.path.join(root_dir, head_dir)
        if not os.path.isdir(head_path):
            continue

        # 提取 head1 和 head2 的值
        if not head_dir.startswith("head1_") or "_head2_" not in head_dir:
            continue
        head1 = int(head_dir.split("_")[1])
        head2 = int(head_dir.split("_")[3])

        sub_path = os.path.join(head_path, sub_dir)
        if not os.path.isdir(sub_path):
            continue

        # 解析 GPU 日志文件
        gpu_before_file = os.path.join(sub_path, 'gpu_memory_before_training.log')
        gpu_after_file = os.path.join(sub_path, 'gpu_memory_after_training.log')
        runtime_file = os.path.join(sub_path, 'runtime.log')
        metrics_file = os.path.join(sub_path, 'metrics.csv')

        if not os.path.exists(gpu_before_file) or not os.path.exists(gpu_after_file):
            continue

        gpu_util_before, memory_used_before = parse_gpu_log(gpu_before_file)
        gpu_util_after, memory_used_after = parse_gpu_log(gpu_after_file)

        if gpu_util_before is None or gpu_util_after is None:
            continue

        # 计算 GPU 和内存变化
        gpu_utilization_changed = gpu_util_after - gpu_util_before
        memory_changed = memory_used_after - memory_used_before

        # 解析运行时间
        elapsed_time = None
        if include_runtime and os.path.exists(runtime_file):
            elapsed_time = parse_runtime_log(runtime_file)

        # 解析指标文件
        rmse, mae, mape = None, None, None
        if include_metrics and os.path.exists(metrics_file):
            rmse, mae, mape = parse_metrics_file(metrics_file)

        # 保存结果
        if include_runtime and include_metrics:
            results.append([head1, head2, gpu_utilization_changed, memory_changed, elapsed_time, rmse, mae, mape])
        elif include_metrics:
            results.append([head1, head2, gpu_utilization_changed, memory_changed, rmse, mae, mape])
        elif include_runtime:
            results.append([head1, head2, gpu_utilization_changed, memory_changed, elapsed_time])
        else:
            results.append([head1, head2, gpu_utilization_changed, memory_changed])

    return results

def save_results_to_csv(results, output_file, include_runtime=False, include_metrics=False):
    """
    将结果保存到 CSV 文件
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        if include_runtime and include_metrics:
            writer.writerow(['head1', 'head2', 'gpu_utilization_changed', 'memory_changed', 'elapsed_time', 'RMSE', 'MAE', 'MAPE'])
        elif include_metrics:
            writer.writerow(['head1', 'head2', 'gpu_utilization_changed', 'memory_changed', 'RMSE', 'MAE', 'MAPE'])
        elif include_runtime:
            writer.writerow(['head1', 'head2', 'gpu_utilization_changed', 'memory_changed', 'elapsed_time'])
        else:
            writer.writerow(['head1', 'head2', 'gpu_utilization_changed', 'memory_changed'])
        writer.writerows(results)

if __name__ == "__main__":
    # 根目录
    root_dir = "head-Results-New"

    # 分别处理 A-C 和 C-A
    for sub_dir in ['A-C', 'C-A']:
        output_file = f"{sub_dir}-headLab.csv"
        results = process_directory_by_subdir(root_dir, sub_dir, include_metrics=True)
        save_results_to_csv(results, output_file, include_metrics=True)
        print(f"{sub_dir} 汇总完成，结果已保存到 {output_file}")