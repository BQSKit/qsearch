import numpy as np
from numpy import matrix, array
from .circuits import *
from warnings import warn
from enum import Enum
import os
import shutil
import sys
import pickle
from multiprocessing import freeze_support
from .compiler import SearchCompiler
from . import solver as scsolver
from .options import Options
from . import logging, checkpoint, utils, gatesets, heuristics, assembler
from time import time


try:
    from mpi4py import MPI
    from .utils import mpi_worker
except ImportError:
    MPI = None

class Project_Status(Enum):
    PROGRESS = 1
    COMPLETE = 2
    NOTBEGUN = 3

class Project:
    def __init__(self, path):
        if MPI is not None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.rank
        else:
            self.comm = None
            self.rank = 0

        if self.rank == 0:
            self.folder = path
            self.name = os.path.basename(os.path.normpath(path))
            self.projfile = os.path.join(path, "qcproject")
            try:
                if not os.path.exists(path):
                    os.mkdir(path)
                with open(self.projfile, "rb") as projfile:
                    self._compilations, self.options = pickle.load(projfile)
                    self.set_defaults()
                    self.logger = logging.Logger(self.options.stdout_enabled, os.path.join(path, "{}-project-log.txt".format(self.name)), self.options.verbosity)
                    self.logger.logprint("Successfully loaded project {}".format(self.name))
                    self.status(logger=self.logger)
            except IOError:
                self._compilations = dict()
                self.options = Options()
                self.set_defaults()
                self.logger = logging.Logger(True, os.path.join(path, "{}-project-log.txt".format(self.name)), verbosity=1)

    def _save(self):
        with open(self.projfile, "wb") as projfile:
            pickle.dump((self._compilations, self.options), projfile)

    def _checkpoint_path(self, name):
        return os.path.join(self.folder, "{}.checkpoint".format(name))

    def add_compilation(self, name, U, options=None, handle_existing=None, **extraargs):
        if name in self._compilations:
            s = self._compilation_status(name)
            if handle_existing == "ignore" or np.array_equal(U, self._compilations[name]["options"].target): # handle the case where this is an attempt to re-enter the same compilation
                if options is None and len(extraargs) == 0:
                    # a "noconflict" method for options could potentially let more cases fall through this return, thereby reducing the number of warnings
                    return
                elif s == Project_Status.PROGRESS or s == Project_Status.COMPLETE:
                    warn("Compilation {} already has made progress.  To change it, reset it first..".format(name), RuntimeWarning, stacklevel=2)
                    return
            elif handle_existing == "overwrite":
                self.remove_compilation(name)
            elif s == Project_Status.PROGRESS or s == Project_Status.COMPLETE:
                warn("A compilation with name {} already exists.  To change it, remove it and then add it again.".format(name), RuntimeWarning, stacklevel=2)
                return
        
        compopt = Options()
        compopt.update(options, **extraargs)
        compopt.target = U
        self._compilations[name] = {"options" : compopt}
        self._save()

    def __setitem__(self, keyword, value):
        if keyword in self.options and self.options[keyword] == value:
            self.options.update(**{keyword:value})
            return # no need to send out a warning if nothing is being changed
        if not keyword in ["verbosity", "stdout_enabled"]: # "safe" keywords here
            for name in self._compilations:
                s = self._compilation_status(name)
                if s == Project_Status.COMPLETE or s == Project_Status.PROGRESS:
                    warn("This project contains compilations which have been completed or have been started.  Please call reset() to clear this progress before changing configurations.", RuntimeWarning, stacklevel=2)
                    return
        self.options.update(**{keyword:value})
        self._save()
        # adjust the logger if relevant
        if keyword == "verbosity":
            self.logger.verbosity = value
        elif keyword == "stdout_enabled":
            self.logger.std_enabled = value

    def configure_compiler_override(self, keyword, value):
        warn("Using this method could result in crashes, infinite loops, or other undefined behavior.  It is safer to reset the project and configure using project[\"keyword\"]=value.  Only use this method if the risk of error is worse than losing intermediate progress.")
        self.options.update(**{keyword:value})
        self._save()

    def __getitem__(self, keyword):
        return self.options[keyword]

    def __delitem__(self, keyword):
        del self.options[keyword]
        # reset logger properties to default if relevant
        if keyword == "verbosity":
            self.logger.verbosity = options.verbosity
        elif keyword == "stdout_enabled":
            self.logger.std_enabled = options.std_enabled

    def configure(self, **dictionary):
        self.options.update(**dictionary)
 
    def reset(self, name=None):
        if name is None:
            [self.reset(n) for n in self._compilations]
        else:
            statefile = self._checkpoint_path(name)
            if os.path.exists(statefile):
                os.remove(statefile)
                self._save()
            cdict = self._compilations[name]
            self._compilations[name] = {"options" : cdict["options"]}
        self._save()


    def remove_compilation(self, name):
        statefile = self._checkpoint_path(name)
        if os.path.exists(statefile):
            os.remove(statefile)
        self._compilations.pop(name)
        self._save()

    def clear(self, name=None):
        if name is None:
            for name in self._compilations:
                statefile = self._checkpoint_path(name)
                checkpoint.delete(statefile)
            self._compilations = dict()
            self._compiler_config = dict()
        else:
            statefile = self._checkpoint_path(name)
            if os.path.exists(statefile):
                os.remove(statefile)
            self._compilations.pop(name)
        self._save()

    def __enter__(self):
        if self.rank == 0:
            return self
        else:
            mpi_worker()
            return self.__exit__(None, None, None)

    def __exit__(self, exc_typ, exc_val, exc_tb):
        if self.rank == 0:
            self.finish()
            return False
        else:
            sys.exit(0)

    def set_defaults(self):
        defaults = {
                "verbosity":1,
                "stdout_enabled":True,
                "blas_threads": None,
                "compiler_class":SearchCompiler,
                }
        self.options.set_defaults(**defaults)

    def run(self, target=None):
        runopt = self.options.copy()
        freeze_support()

        self.logger.logprint("Started running project {}".format(self.name))
        blas_threads = self.options.blas_threads

        CompilerClass = self.options.compiler_class
        compiler = CompilerClass(runopt)
        self.status(logger=self.logger)
        for name in self._compilations:
            cdict = self._compilations[name]

            statefile = self._checkpoint_path(name)
            if self._compilation_status(name) == Project_Status.COMPLETE:
                continue
            sublogger = logging.Logger(self.options.stdout_enabled, os.path.join(self.folder, "{}-log.txt".format(name)), self.options.verbosity)
            runopt.logger = sublogger
            self.logger.logprint("Starting compilation of {}".format(name))
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                starttime = time()
                structure, vector = compiler.compile(runopt.updated(cdict["options"]))
            else:
                with threadpool_limits(limits=blas_threads, user_api='blas'):
                    starttime = time()
                    structure, vector = compiler.compile(runopt.updated(cdict["options"]))
            endtime = time()
            self.logger.logprint("Finished compilation of {}".format(name))
            cdict["structure"] = structure
            cdict["vector"] = vector
            cdict["time"] = endtime - starttime
            self._compilations[name] = cdict
            self._save()
            self.logger.logprint("Recorded results from compilation.", verbosity=2)

            checkpoint.delete(statefile)
            self.logger.logprint("Deleted checkpoint file.", verbosity=2)
            self.status(logger=self.logger)
        self.logger.logprint("Finished running project {}".format(self.name))

    def complete(self):
        return self._overall_status() == Project_Status.COMPLETE

    def finish(self):
        if MPI is not None:
            self.comm.bcast(True, root=0)

    def status(self, name=None, logger=None):
        namelist = [name] if name else self._compilations
        for n in namelist:
            s = self._compilation_status(n)
            msg = ""
            if s == Project_Status.COMPLETE:
                msg = "Complete!"
            elif s == Project_Status.PROGRESS:
                msg = "In progress..."
            elif s == Project_Status.NOTBEGUN:
                msg = "Not started."
            elif s == Project_Status.DEBUGING:
                msg = "Debug."
            if logger is None:
                print("{}\t{}".format(n,msg))
            else:
                logger.logprint("{}\t{}".format(n,msg))

    @property
    def compilations(self):
        return list(self._compilations.keys())

    def _compilation_status(self, name):
        cdict = self._compilations[name]
        if os.path.exists(self._checkpoint_path(name)):
            return Project_Status.PROGRESS
        elif "structure" in cdict and "vector" in cdict:
            return Project_Status.COMPLETE
        else:
            return Project_Status.NOTBEGUN

    def _overall_status(self):
        complete = True
        started = False
        if name:
            return self._compilation_status(name)
        for name in self._compilations:
            s = self._compilation_status(name)
            if s != Project_Status.COMPLETE:
                complete = False
            if s != Project_Status.NOTBEGUN:
                started = True
        if complete:
            return Project_Status.COMPLETE
        elif started:
            return Project_Status.PROGRESS
        else:
            return Project_Status.NOTBEGUN

    def get_result(self, name):
        cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
            return None, None

        return cdict["structure"], cdict["vector"]

    def get_target(self, name):
        cdict = self._compilations[name]
        return cdict["options"].target

    def get_time(self, name):
        cdict = self._compilations[name]
        if "time" in cdict:
            return cdict["time"]
        else:
            return None

    def verify_result(self, name):
        cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("The compilation {} has not been completed.  Please run the project to finish the compilation.")
            return
        original = cdict["options"].target
        final = cdict["structure"].matrix["vector"]
        print("Comparison of target and implemented unitaries:")
        if "error_func" in self._compiler_config:
            print("error_func: {}".format(self._compiler_config["error_func"](original, final)))
        print("matrix_distance_squared: {}".format(utils.matrix_distance_squared(original, final)))

    def assemble(self, name, language=assembler.ASSEMBLY_IBMOPENQASM, write_location=None):
        cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
            return None, None

        out = assembler.assemble(cdict["structure"], cdict["vector"], language, write_location)
        if write_location == None:
            return out

