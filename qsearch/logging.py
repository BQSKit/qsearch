"""
This module defines the Logger class, which is used to control and automate the logging of messages to stdout and to files.
"""
import datetime
import os.path as path
import numpy as np

class Logger:
    """This class is used to control what level of mesages get printed, and to where."""
    def __init__(self, stdout_enabled=False, output_file=None, verbosity=1):
        self.stdout_enabled=stdout_enabled
        self.output_file=output_file
        self.verbosity=verbosity

    def logprint(self, string, verbosity=1):
        """This function logs the specified string according to the specified options."""
        if verbosity > self.verbosity:
            return # ignore print requests for a higher verbosity than our current setting
        if self.stdout_enabled:
            print(string)
        if self.output_file is not None:
            with open(self.output_file, "a") as f:
                f.write(str(datetime.datetime.now()) + " > \t" + str(string) + "\n")


