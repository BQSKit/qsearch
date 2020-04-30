import datetime
import os.path as path
import numpy as np

class Logger:
    def __init__(self, stdout_enabled=False, output_file=None, verbosity=1):
        self.stdout_enabled=stdout_enabled
        self.output_file=output_file
        self.verbosity=verbosity

    def logprint(self, string, verbosity=1):
        if verbosity > self.verbosity:
            return # ignore print requests for a higher verbosity than our current setting
        if self.stdout_enabled:
            print(string)
        if self.output_file is not None:
            with open(self.output_file, "a") as f:
                f.write(str(datetime.datetime.now()) + " > \t" + str(string) + "\n")


