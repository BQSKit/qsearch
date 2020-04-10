import numpy as np
from numpy import matrix, array
from .circuits import *
from warnings import warn
from enum import Enum
import os
import shutil
import pickle
from .compiler import SearchCompiler
from .solver import default_solver
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
                self.status()
        except IOError:
            self._compilations = dict()
            self._compiler_config = dict()

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
        for name in self._compilations:
            s = self._compilation_status(name)
            if s == Project_Status.COMPLETE or s == Project_Status.PROGRESS:
                warn("This project contains compilations which have been completed or have been started.  Please call reset() to clear this progress before changing configurations.", RuntimeWarning, stacklevel=2)
                return
        self._compiler_config[keyword] = value
        self._save()

    def configure_compiler_override(self, keyword, value):
        warn("Using this method could result in crashes, infinite loops, or other undefined behavior.  It is safer to reset the project and configure using project[\"keyword\"]=value.  Only use this method if the risk of error is worse than losing intermediate progress.")
        self._compiler_config[keyword] = value
        self._save()

    def __getitem__(self, keyword):
        return self._compiler_config[keyword]

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
        print("Started running project {}".format(self.name))
        threshold = self._config("threshold", 1e-10)
        gateset = self._config("gateset", gatesets.QubitCNOTLinear())
        error_func = self._config("error_func", utils.matrix_distance_squared)
        heuristic = heuristics.astar
        d = self._config("d", 2)
        max_dits = int(np.log(max([self._compilations[name][0].shape[0] for name in self._compilations]))/np.log(2))
        if "search_type" in self._compiler_config:
            st = self._compiler_config["search_type"]
            if st == "breadth":
                heuristic = heuristics.breadth
            elif st == "greedy":
                heuristic = heuristics.greedy

        heuristic = self._config("heuristic", heuristic)
        solver = self._config("solver", default_solver(gateset, max_dits, error_func))
        beams = self._config("beams", -1)
        depthlimit = self._config("depth", None)
        blas_threads = self._config("blas_threads", None)
        compiler = SearchCompiler(threshold=threshold, gateset=gateset, error_func=error_func, heuristic=heuristic, solver=solver, beams=beams)
        self.status()
        for name in self._compilations:
            U, cdict = self._compilations[name]

            statefile = self._checkpoint_path(name)
            if self._compilation_status(name) == Project_Status.COMPLETE:
                continue

            logging.output_file = os.path.splitext(self.projfile)[0] + "-{}".format(name)
            logging.logprint("Starting compilation of {}".format(name))
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                starttime = time()
                result, structure, vector = compiler.compile(U, depth=depthlimit, statefile=statefile)
            else:
                with threadpool_limits(limits=blas_threads, user_api='blas'):
                    starttime = time()
                    result, structure, vector = compiler.compile(U, depth=depthlimit, statefile=statefile)
            endtime = time()
            logging.logprint("Finished compilation of {}".format(name))
            cdict["result"] = result
            cdict["structure"] = structure
            cdict["vector"] = vector
            cdict["time"] = endtime - starttime
            self._compilations[name] = (U, cdict)
            self._save()
            logging.logprint("Recorded results from compilation.")

            checkpoint.delete(statefile)
            logging.logprint("Deleted checkpoint file.")
            self.status()
        print("Finished running project {}".format(self.name))

    def complete(self):
        return self._overall_status() == Project_Status.COMPLETE

    def status(self, name=None):
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
            print("{}\t{}".format(n,msg))

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

    def verify_result(self, name, count=1000):
        original, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("The compilation {} has not been completed.  Please run the project to finish the compilation.")
            return

        final = cdict["result"]
        maxs, total, mins = sc.utils.random_vector_evaluation(original, final, count)

        print("Max: {}%\nAverage: {}%\nMin: {}%\n".format(maxs*100.0, total*100.0, mins*100.0))


    def assemble(self, name, language=assembler.ASSEMBLY_IBMOPENQASM, write_location=None):
        _, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
            return None, None

        out = assembler.assemble(cdict["structure"], cdict["vector"], language, write_location)
        if write_location == None:
            return out

