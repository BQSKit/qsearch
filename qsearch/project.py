"""
This module provides a wrapper that makes it easier to interface with the rest of Qsearch.
"""
import numpy as np
from numpy import matrix, array
from warnings import warn
from enum import Enum
import os
import shutil
import sys
import pickle
from multiprocessing import freeze_support
from .compiler import SearchCompiler
from . import solvers as scsolver
from .options import Options
from .defaults import standard_defaults, standard_smart_defaults, objectives
from . import logging, utils, gatesets, heuristics, assemblers
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
    """The project class wraps most of the functionality of Qsearch as intended to help manage working with Qsearch."""
    def __init__(self, path, use_mpi=False):
        if MPI is not None and use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.rank
        else:
            self.comm = None
            self.rank = 0

        if self.rank == 0:
            self.aborted = False
            self.folder = path
            self.name = os.path.basename(os.path.normpath(path))
            self.projfile = os.path.join(path, "qcproject")
            try:
                if not os.path.exists(path):
                    os.mkdir(path)
                with open(self.projfile, "rb") as projfile:
                    self._compilations, self.options = pickle.load(projfile)
                    self.logger = logging.Logger(self.options.stdout_enabled, os.path.join(path, "{}-project-log.txt".format(self.name)), self.options.verbosity)
                    self.logger.logprint("Successfully loaded project {}".format(self.name))
                    self.status(logger=self.logger)
            except IOError:
                self._compilations = dict()
                self.options = Options()
                self.set_defaults()
                self.set_smart_defaults()
                self.logger = logging.Logger(True, os.path.join(path, "{}-project-log.txt".format(self.name)), verbosity=1)

    def _save(self):
        with open(self.projfile, "wb") as projfile:
            pickle.dump((self._compilations, self.options), projfile)

    def _checkpoint_path(self, name):
        return os.path.join(self.folder, "{}.checkpoint".format(name))

    def add_compilation(self, name, U, options=None, handle_existing=None, **extraargs):
        """
        Adds a unitary to be compiled.

        Args:
            name : A name for this unitary.  Must be unique in this Project.
            U : The unitary to be compiled, in the form of a numpy ndarray with dtype="complex128"
            handle_existing : A variable which defines how to behave if a compilation with the given name already exists.  If it is set to "ignore", it will simply return without doing anything.  If it is set to "overwrite", it will overwrite the previous entry.  If it is set to the default of None, it will offer a warning asking the user to remove and re-add the compilation.
            options : The options passed to this function will be used only when this compilation is run.
            extraargs : The extraargs passed to this function will be used only when this compilation is run.
        """
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
        compopt = Options(statefile=self._checkpoint_path(name))
        compopt.update(options, **extraargs)
        compopt.target = U
        self._compilations[name] = {"options" : compopt}
        self._save()

    def __setitem__(self, keyword, value):
        try:
            if keyword in self.options and self.options[keyword] == value:
                self.options.update(**{keyword:value})
                return # no need to send out a warning if nothing is being changed
        except ValueError:
            self.options.update(keyword=value)
            return
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
        """An unsafe method that allows the user to set global Project Options even if there is existing work."""
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
        """Adds multiple options to the global Project Options at once."""
        self.options.update(**dictionary)
 
    def reset(self, name=None):
        """Resets a Project, removing any work done but not the initial configurations.

        Args:
            name: Optionally specify a particular compilation by name to reset
        """
        if name is None:
            [self.reset(n) for n in self._compilations]
        else:
            cdict = self._compilations[name]
            self.get_options(name).checkpoint.delete()
            self._compilations[name] = {"options" : cdict["options"]}
        self._save()

    def remove_compilation(self, name):
        """Removes a compilation from a Project.

        Args:
            name: The name of the compilation to remove
        """
        warn("remove_compilation(name) is deprecated and will be removed in a future release.  Use clear(name) instead.", DeprecationWarning, stacklevel=2)
        self.get_options(name).checkpoint.delete()
        cdict = self._compilations.pop(name)
        self._save()

    def clear(self, name=None):
        """Clears a Project, reverting it to a state similar to a newly created Project.

        Args:
            name: Optionally specify a particular compilation by name to clear
        """
        if name is None:
            for name in self._compilations:
                self.get_options(name).checkpoint.delete()
            self._compilations = dict()
            self._compiler_config = dict()
        else:
            self.get_options(name).checkpoint.delete()
            cdict = self._compilations.pop(name)
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

    def set_defaults(self, defaults=standard_defaults):
        """Updates the Project Options with the standard defaults from defaults.py, or a provided dictionary."""
        self.options.set_defaults(**defaults)

    def set_smart_defaults(self, smart_defaults=standard_smart_defaults):
        """Updates the Project Options with the standard smart_defaults from defaults.py, or a provided dictionary"""
        self.options.set_smart_defaults(**smart_defaults)

    def run(self):
        """Runs all of the compilations in the Project."""
        freeze_support()
        self.aborted = False
        self.logger.logprint("Started running project {}".format(self.name))
        self.status(logger=self.logger)
        for name in self._compilations:
            cdict = self._compilations[name]

            runopt = self.get_options(name)
            blas_threads = runopt.blas_threads
            CompilerClass = runopt.compiler_class
            compiler = CompilerClass(runopt)

            if self._compilation_status(name) == Project_Status.COMPLETE:
                continue
            sublogger = logging.Logger(runopt.stdout_enabled, os.path.join(self.folder, "{}-log.txt".format(name)), runopt.verbosity)
            runopt.logger = sublogger
            self.logger.logprint("Starting compilation of {}".format(name))
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                starttime = time()
                try:
                    result = compiler.compile(runopt)
                except KeyboardInterrupt:
                    self.aborted = True
                    self.logger.logprint("\nStopping due to Ctrl+C...\n")
                    self.status()
                    return
            else:
                with threadpool_limits(limits=blas_threads, user_api='blas'):
                    starttime = time()
                    try:
                        result = compiler.compile(runopt)
                    except KeyboardInterrupt:
                        self.aborted = True
                        self.logger.logprint("\nStopping due to Ctrl+C...\n")
                        self.status()
                        return
            endtime = time()
            self.logger.logprint("Finished compilation of {}".format(name))
            cdict.update(**result)
            cdict["time"] = endtime - starttime
            self._compilations[name] = cdict
            self._save()
            self.logger.logprint("Recorded results from compilation.", verbosity=2)

            runopt.checkpoint.delete()
            self.logger.logprint("Deleted checkpoint file.", verbosity=2)
            self.status(logger=self.logger)
        self.logger.logprint("Finished running project {}".format(self.name))

    def post_process(self, postprocessor, name=None, options=None, **xtraargs):
        """Post-processes the specified compilation, or all compilations if name is None, using the specified postprocessor.

        Args:
            postprocessor: The qsearch.post_processing.PostProcessor to run on the compilation or project
            name : Optionally specify a particular compilation by name to reset
            options : Options to pass to the qsearch.post_processing.PostProcessor passed in `postprocessor`
            extraargs : Extra arguments passed as options to the qsearch.post_processing.PostProcessor passed in `postprocessor`
        """
        if not self.aborted:
            names = [name] if name else self._compilations
            for name in names:
                self.logger.logprint("Started postprocessing of {}".format(name))
                cdict = self._compilations[name]
                finaloptions = self.options.updated(cdict["options"]).updated(options, **xtraargs)
                result = postprocessor.post_process_circuit(cdict, finaloptions)
                cdict.update(**result)
                self._compilations[name] = cdict
                self.logger.logprint("Finished postprocessing of {}".format(name))
            self._save()
            
    def complete(self):
        """Returns a True if all compilations in the Project have finished and False otherwise."""
        return self._overall_status() == Project_Status.COMPLETE

    def finish(self):
        """Called when done running compilations in order to end MPI tasks."""
        if MPI is not None and self.comm is not None:
            self.comm.bcast(True, root=0)

    def status(self, name=None, logger=None):
        """Prints a status update on how much of a Project has finished running.

        Args:
            name: Optionally specify which compilation to check the status of
        """
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
        """The list of names corresponding to compilations on this Project."""
        return list(self._compilations.keys())

    def _compilation_status(self, name):
        cdict = self._compilations[name]
        if os.path.exists(self._checkpoint_path(name)):
            return Project_Status.PROGRESS
        elif "structure" in cdict and "parameters" in cdict:
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
        """Get the result of a compilation.

        Args:
            name: The name of the compilation to get the result dictionary from

        Returns:
            dict: The result dictionary for a finished compilation.  Usually this contains the entries "structure", a Gate, and "parameters", an array of real number parameters.
        """
        cdict = self._compilations[name]
        if not "structure" in cdict or not "parameters" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
        return cdict

    def get_target(self, name):
        """Get the target unitary of a compilation.

        Args:
            name: The name of the compilation to get the target from

        Returns:
            np.ndarray: The target unitary of the compilation
        """
        cdict = self._compilations[name]
        return cdict["options"].target

    def get_time(self, name):
        """Get the runtime that it took to run a compilation.

        Args:
            name: The name of the compilation to get the runtime of

        Returns:
            float: The number of seconds the compilation took
        """
        cdict = self._compilations[name]
        if "time" in cdict:
            return cdict["time"]
        else:
            return None

    def get_options(self, name=None):
        """Get the qsearch.options.Options object from a compilation of project

        Args:
            name: Optionally pass the name of the compilation to get the qsearch.options.Options object from

        Returns:
            qsearch.options.Options: the requested options object
        """
        if name is None:
            return self.options
        else:
            return self.options.updated(self._compilations[name]["options"])

    def assemble(self, name, options=None, **xtraargs):
        """Assembles a compilation using the Assembler specified as assembler in the Options.
        Args:
            name: The compilation to assemble
            options: Contains the qsearch.assemblers.Assembler to use in assembly

        Returns:
            str: The resulting assembled code
        """
        options = self.options.updated(options, **xtraargs)
        cdict = self._compilations[name]
        if not "structure" in cdict or not "parameters" in cdict:
            #TODO change this to a logprint
            print("This compilation has not been completed.  please run the project to complete the compilation.")
            return None

        out = options.assembler.assemble(cdict, options)
        if options.write_location is not None:
            with open(options.write_location, "w") as wfile:
                wfile.write(out)
        else:
            return out

