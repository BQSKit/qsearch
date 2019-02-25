import datetime

stdout_enabled = True
output_file = "pylog.txt"

def logprint(string):
    if stdout_enabled:
        print(string)
    if output_file:
        with open(output_file, "a") as f:
            f.write(str(datetime.datetime.now()) + " > \t" + str(string) + "\n")


