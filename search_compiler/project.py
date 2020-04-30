import numpy as np
from numpy import matrix, array
from .circuits import *
from warnings import warn
from enum import Enum
import os
import shutil
import pickle
from .compiler import SearchCompiler
from . import solver as scsolver
from . import logging, checkpoint, utils, gatesets, heuristics, assembler
from time import time

class Project_Status(Enum):
    PROGRESS = 1
    COMPLETE = 2
    NOTBEGUN = 3

class Project:
    def __init__(self, path):
        self.folder = path
        self.name = os.path.basename(os.path.normpath(path))
        self.projfile = os.path.join(path, "qcproject")
        try:
            if not os.path.exists(path):
                os.mkdir(path)
            with open(self.projfile, "rb") as projfile:
                self._compilations, self._compiler_config = pickle.load(projfile)
                self.logger = logging.Logger(self._config("stdout_enabled", True), os.path.join(path, "{}-project-log.txt".format(self.name)), self._config("verbosity", 1))
                self.logger.logprint("Successfully loaded project {}".format(self.name))
                self.status(logger=self.logger)
        except IOError:
            self._compilations = dict()
            self._compiler_config = dict()
            self.logger = logging.Logger(True, os.path.join(path, "{}-project-log.txt".format(self.name)), verbosity=1)

    def _save(self):
        with open(self.projfile, "wb") as projfile:
            pickle.dump((self._compilations, self._compiler_config), projfile)

    def _config(self, keyword, default):
        if keyword in self._compiler_config:
            return self._compiler_config[keyword]
        else:
            return default

    def _checkpoint_path(self, name):
        return os.path.join(self.folder, "{}.checkpoint".format(name))
        return os.path.splitext(self.projfile)[0] + "-{}.checkpoint".format(name)

    def add_compilation(self, name, U, handle_existing=None):
        if name in self._compilations:
            s = self._compilation_status(name)
            if handle_existing == "ignore" or np.array_equal(U, self._compilations[name][0]): # ignore if the ignore flag is specified or if the matrix is the same as the already existing one
                return
            elif handle_existing == "overwrite":
                self.remove_compilation(name)
            elif s == Project_Status.PROGRESS or s == Project_Status.COMPLETE:
                warn("A compilation with name {} already exists.  To change it, remove it and then add it again.".format(name), RuntimeWarning, stacklevel=2)
                return
        
        self._compilations[name] = (U, dict())
        self._save()

    def __setitem__(self, keyword, value):
        if keyword in self._compiler_config and self._compiler_config[keyword] == value:
            return # no need to send out a warning if nothing is being changed
        if not keyword in ["verbosity", "stdout_enabled"]: # "safe" keywords here
            for name in self._compilations:
                s = self._compilation_status(name)
                if s == Project_Status.COMPLETE or s == Project_Status.PROGRESS:
                    warn("This project contains compilations which have been completed or have been started.  Please call reset() to clear this progress before changing configurations.", RuntimeWarning, stacklevel=2)
                    return
        self._compiler_config[keyword] = value
        self._save()
        # adjust the logger if relevant
        if keyword == "verbosity":
            self.logger.verbosity = value
        elif keyword == "stdout_enabled":
            self.logger.std_enabled = value

    def configure_compiler_override(self, keyword, value):
        warn("Using this method could result in crashes, infinite loops, or other undefined behavior.  It is safer to reset the project and configure using project[\"keyword\"]=value.  Only use this method if the risk of error is worse than losing intermediate progress.")
        self._compiler_config[keyword] = value
        self._save()

    def __getitem__(self, keyword):
        return self._compiler_config[keyword]

    def __delitem__(self, keyword):
        del self._compiler_config[keyword]
        # reset logger properties to default if relevant
        if keyword == "verbosity":
            self.logger.verbosity = 1
        elif keyword == "stdout_enabled":
            self.logger.std_enabled = True

    def configure(self, dictionary):
        for key in dictionary:
            self[key] = dictionary[key]
 
    def reset(self, name=None):
        if name is None:
            [self.reset(n) for n in self._compilations]
        else:
            statefile = self._checkpoint_path(name)
            if os.path.exists(statefile):
                os.remove(statefile)
                self._save()
            U, cdict = self._compilations[name]
            cdict.pop("vector", None)
            cdict.pop("structure", None)
            self._compilations[name] = (U, cdict)
        self._save()


    def remove_compilation(self, name):
        statefile = self._checkpoint_path(name)
        if os.path.exists(statefile):
            os.remove(statefile)
        self._compilations.pop(name)
        self._save()

    def clear(self):
        for name in self._compilations:
            statefile = self._checkpoint_path(name)
            checkpoint.delete(statefile)
        self._compilations = dict()
        self._compiler_config = dict()
        self._save()

    def run(self, target=None):
        self.logger.logprint("Started running project {}".format(self.name))
        threshold = self._config("threshold", 1e-10)
        gateset = self._config("gateset", gatesets.QubitCNOTLinear())
        max_dits = int(np.log(max([self._compilations[name][0].shape[0] for name in self._compilations]))/np.log(2))
        error_func = self._config("error_func", utils.matrix_distance_squared)
        error_jac = self._config("error_jac", None)
        if error_jac is None:
            if error_func == utils.matrix_distance_squared:
                error_jac = utils.matrix_distance_squared_jac
            elif error_func == utils.matrix_residuals:
                error_jac = utils.matrix_residuals_jac
        eval_func = self._config("eval_func", None)
        if eval_func is None:
            if error_func == utils.matrix_residuals:
                eval_func = utils.matrix_distance_squared
            else:
                eval_func = error_func

        solver = self._config("solver", None)
        if solver is None:
            solver = scsolver.default_solver(gateset, max_dits, error_func, error_jac, self.logger)
            if type(solver) == scsolver.LeastSquares_Jac_Solver or type(solver) == scsolver.LeastSquares_Jac_SolverNative:
                # if default_solver made the call to switch to LeastSquares, then these functions should be assigned to these values
                error_func = utils.matrix_residuals
                error_jac = utils.matrix_residuals_jac
                eval_func = utils.matrix_distance_squared

        d = self._config("d", 2)
        heuristic=heuristics.astar
        if "search_type" in self._compiler_config:
            st = self._compiler_config["search_type"]
            if st == "breadth":
                heuristic = heuristics.breadth
            elif st == "greedy":
                heuristic = heuristics.greedy
        heuristic = self._config("heuristic", heuristic)
        beams = self._config("beams", -1)
        depthlimit = self._config("depth", None)
        blas_threads = self._config("blas_threads", None)
        verbosity = self._config("verbosity", 1)
        stdout_enabled = self._config("stdout_enabled", True)

        compiler = SearchCompiler(threshold=threshold, gateset=gateset, error_func=error_func, error_jac=error_jac, eval_func=eval_func, heuristic=heuristic, solver=solver, beams=beams)
        self.status(logger=self.logger)
        for name in self._compilations:
            U, cdict = self._compilations[name]

            statefile = self._checkpoint_path(name)
            if self._compilation_status(name) == Project_Status.COMPLETE:
                continue
            sublogger = logging.Logger(stdout_enabled, os.path.join(self.folder, "{}-log.txt".format(name)), verbosity)
            self.logger.logprint("Starting compilation of {}".format(name))
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                starttime = time()
                structure, vector = compiler.compile(U, depth=depthlimit, statefile=statefile, logger=sublogger)
            else:
                with threadpool_limits(limits=blas_threads, user_api='blas'):
                    starttime = time()
                    structure, vector = compiler.compile(U, depth=depthlimit, statefile=statefile, logger=sublogger)
            endtime = time()
            self.logger.logprint("Finished compilation of {}".format(name))
            cdict["structure"] = structure
            cdict["vector"] = vector
            cdict["time"] = endtime - starttime
            self._compilations[name] = (U, cdict)
            self._save()
            self.logger.logprint("Recorded results from compilation.", verbosity=2)

            checkpoint.delete(statefile)
            self.logger.logprint("Deleted checkpoint file.", verbosity=2)
            self.status(logger=self.logger)
        self.logger.logprint("Finished running project {}".format(self.name))

    def complete(self):
        return self._overall_status() == Project_Status.COMPLETE

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
        _, cdict = self._compilations[name]
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
        _, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
            return None, None

        return cdict["structure"], cdict["vector"]

    def get_target(self, name):
        U, _ = self._compilations[name]
        return U

    def get_time(self, name):
        _, cdict = self._compilations[name]
        if "time" in cdict:
            return cdict["time"]
        else:
            return None

    def verify_result(self, name):
        original, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("The compilation {} has not been completed.  Please run the project to finish the compilation.")
            return

        final = cdict["structure"].matrix["vector"]
        print("Comparison of target and implemented unitaries:")
        if "error_func" in self._compiler_config:
            print("error_func: {}".format(self._compiler_config["error_func"](original, final)))
        print("matrix_distance_squared: {}".format(utils.matrix_distance_squared(original, final)))

    def assemble(self, name, language=assembler.ASSEMBLY_IBMOPENQASM, write_location=None):
        _, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
            return None, None

        out = assembler.assemble(cdict["structure"], cdict["vector"], language, write_location)
        if write_location == None:
            return out

