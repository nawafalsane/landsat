import subprocess
import os
import glob


xmin, ymin, xmax, ymax = 322160.04513999925, 4691046.776180349, 330043.30301649845, 4696916.347295563

for filepath in glob.iglob('./raw_files/*'):
    name = filepath.split('/')[-1]
    name = name.replace('\n', '')
    name = 'cliped_' + name 
    if os.path.exists(name):
        print("File ", name," exists")
        continue
    else:
        completed = subprocess.run(['rio','clip', filepath, name, '--bounds', xmin, ymin, xmax, ymax])
        print('returncode:', completed.returncode)
    print("File ", name, " cliped")
