import subprocess
import os

with open('landsat8_link.txt', 'r') as file:
    for line in file:
        name = line.split("/")[-1]
        if os.path.exists(name):
            print("File ", name," exists")
            continue
        else:
            completed = subprocess.run(['gsutil', 'cp', line, name])
            print('returncode:', completed.returncode)
        print("File ", name, " Downloaded")