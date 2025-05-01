import subprocess

def run_task(task_name, module):
    print(f"Starting {task_name}...")
    try:
        subprocess.run(["python3", "-c", f"from forLines import {module}; {module}.main()"], check=True)
        print(f"Finished {task_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {task_name}: {e}")
        exit(1)

if __name__ == "__main__":
    tasks = [
        # ("Data Excavation", "forLine2_dataExcavate"),
        # ("Data Cleaning", "forLine3_dataClean"),
        ("Direction Depart", "forLine4_directionDepart"),
        ("Direction Sort", "forLine5_directionSort"),
        ("IP Statistics", "forLine6_ipStatistics"),
        ("Day Line", "forLine7_dayLine"),
        ("Day Line Constant","8_dayLineConstant")
    ]

    for task_name, module in tasks:
        run_task(task_name, module)