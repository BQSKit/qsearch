import datetime
import os.path as path
import numpy as np
 
stdout_enabled = True
output_file = "pylog.txt"

def logstandard(message, heuristic, value, depth):
    logprint("{}\tH: {}\tV: {}\tD: {}".format(message, np.around(heuristic, 5), np.around(value, 5), depth))

def logprint(string, custom=None):
    if stdout_enabled and custom is None:
        print(string)
    if custom is None:
        custom = "pylog"
    if output_file:
        with open(output_file + "-" + custom+".txt", "a") as f:
            f.write(str(datetime.datetime.now()) + " > \t" + str(string) + "\n")


