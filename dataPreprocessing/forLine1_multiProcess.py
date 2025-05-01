'''
脚本通过24个进程同时遍历桌面上subCap文件夹下的所有子文件夹，
对每个子文件夹下的pcap文件进行分析，提取出每个pcap文件的起始时间和结束时间，
并将pcap文件按照起始时间和结束时间的日期进行分类，存放到byDate文件夹下的对应日期文件夹中。
同时，脚本会生成一个CSV文件，记录每个pcap文件的起始时间和结束时间，以及存放的日期文件夹。

'''
import os
import shutil
import csv
from scapy.all import rdpcap
import time
from multiprocessing import Pool
from tqdm import tqdm

PATH_TOTAL = '../subCap'
PATH_DATE = '../byDate'
CSV_FILE_TEMPLATE = 'time_distribution(subdir_{0}).csv'

def analyze_pcap(file_path):
    packets = rdpcap(file_path)
    if not packets:
        return None, None
    start_time = float(packets[0].time)
    end_time = float(packets[-1].time)
    return start_time, end_time

def create_date_folder(start_time, end_time):
    start_date = time.strftime('%Y-%m-%d', time.localtime(start_time))
    end_date = time.strftime('%Y-%m-%d', time.localtime(end_time))
    if start_date == end_date:
        folder_name = start_date
    else:
        folder_name = f"{start_date}-{end_date}"
    folder_path = os.path.join(PATH_DATE, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def format_timestamp(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

def process_subdir(subdir):
    subdir_path = os.path.join(PATH_TOTAL, subdir)
    csv_file_path = CSV_FILE_TEMPLATE.format(subdir)
    
    # Check if CSV file exists and read existing pcap names
    existing_pcap_names = set()
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header
            for row in csv_reader:
                existing_pcap_names.add(row[1])
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['subdir', 'pcap_name', 'start_time', 'end_time'])
        if os.path.isdir(subdir_path):
            pcap_files = [f for f in os.listdir(subdir_path) if f.endswith('.pcap')]
            for pcap_file in tqdm(pcap_files, desc=f"Processing {subdir}"):
                if pcap_file in existing_pcap_names:
                    continue
                pcap_path = os.path.join(subdir_path, pcap_file)
                start_time, end_time = analyze_pcap(pcap_path)
                if start_time and end_time:
                    folder_path = create_date_folder(start_time, end_time)
                    # Check if the file already exists in the destination folder
                    dest_file_path = os.path.join(folder_path, pcap_file)
                    if not os.path.exists(dest_file_path):
                        shutil.copy(pcap_path, folder_path)
                    formatted_start_time = format_timestamp(start_time)
                    formatted_end_time = format_timestamp(end_time)
                    csv_writer.writerow([subdir, pcap_file, formatted_start_time, formatted_end_time])
                    # print(f"Written to CSV: {subdir}, {pcap_file}, {formatted_start_time}, {formatted_end_time}")

def main():
    subdirs = [d for d in os.listdir(PATH_TOTAL) if os.path.isdir(os.path.join(PATH_TOTAL, d))]
    with Pool(processes=24) as pool:
        pool.map(process_subdir, subdirs)

if __name__ == '__main__':
    main()