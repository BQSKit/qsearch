from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
import sys

def quit(err):
    print(err)
    sys.exit(1)
avg = '--avg' in sys.argv
if avg:
    sys.argv.remove('--avg')
args = [Path(p) for p in sys.argv[1:]]
logs = {}
for dir in args:
    if not dir.exists():
        quit("Arguments must be project directories")
    for file in dir.iterdir():
        if not file.suffix:
            continue
        name = str(file).split('-')[-2]
        if name not in logs:
            logs[name] = {str(dir): file}
        else:
            logs[name][str(dir)] = file

def get_time(file):
    with open(file) as f:
        start, end = [line.split('>')[0].strip() for line in f.readlines() if 'Starting' in line or 'Finished compilation of' in line]
    start_t = datetime.fromisoformat(start)
    end_t = datetime.fromisoformat(end)
    return end_t - start_t

for bench, files in logs.items():
    print(f"{bench}:")
    if not avg:
        for run in files:
            print(run, get_time(files[run]))
    else:
        print(sum([get_time(run) for run in files.values()], timedelta())/len(args))
