from threadpoolctl import threadpool_limits
import numpy as np
from numpy import matrix, array
from .circuits import *
from warnings import warn
import os
import shutil
import pickle
from .compiler import SearchCompiler
from .solver import default_solver
from . import logging, checkpoint, utils, gatesets, heuristics, assembler

PROJECT_STATUS_PROGRESS = 1
PROJECT_STATUS_COMPLETE = 2
PROJECT_STATUS_NOTBEGUN = 3
PROJECT_STATUS_DEBUGING = 4

class Project:
    def __init__(self, path, debug=False):
        self.folder = path
        self.projfile = os.path.join(path, "qcproject")
        try:
            if not os.path.exists(self.projfile):
                os.mkdir(self.projfile)

            with open(self.projfile, "rb") as projfile:
                self._compilations, self._compiler_config = pickle.load(projfile)
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

    def add_compilation(self, name, U, debug=False, handle_existing=None):
        if name in self._compilations:
            s = self.compilation_status(name)
            if handle_existing == "ignore":
                return
            elif handle_existing == "overwrite":
                self.remove_compilation(name)
            elif s == PROJECT_STATUS_PROGRESS or s == PROJECT_STATUS_COMPLETE:
                warn("A compilation with name {} already exists.  To change it, remove it and then add it again.".format(name), RuntimeWarning, stacklevel=2)
                return
        
        self._compilations[name] = (U, {"debug" : debug})
        self._save()

    def configure_compiler(self, keyword, value, force=False):
        if not force:
            for name in self._compilations:
                s = self.compilation_status(name)
                if s == PROJECT_STATUS_COMPLETE or s == PROJECT_STATUS_PROGRESS:
                    warn("This project contains compilations which have been completed or have been started.  Changing the compiler configuration will delete the existing progress.  Call configure_compiler with the 'force' option set to True to reset these compilations and reconfigure the compiler.", RuntimeWarning, stacklevel=2)
                    self.status()
                    return
        else:
            for name in self._compilations:
                U, cdict = self._compilations[name]
                self.add_compilation(name, U, debug=cdict["debug"], handle_existing="overwrite")

        self._compiler_config[keyword] = value
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
        threshold = self._config("threshold", 1e-10)
        gateset = self._config("gateset", gatesets.QubitCNOTLinear())
        error_func = self._config("error_func", utils.matrix_distance_squared)
        heuristic = heuristics.astar
        d = self._config("d", 2)
        if "search_type" in self._compiler_config:
            st = self._compiler_config["search_type"]
            if st == "breadth":
                heuristic = heuristics.breadth
            elif st == "greedy":
                heuristic = heuristics.greedy

        heuristic = self._config("heuristic", heuristic)
        solver = self._config("solver", default_solver())
        beams = self._config("beams", 1)
        blas_threads = self._config("blas_threads", None)
        compiler = SearchCompiler(threshold=threshold, d=d, gateset=gateset, error_func=error_func, heuristic=heuristic, solver=solver, beams=beams)
        self.status()
        for name in self._compilations:
            U, cdict = self._compilations[name]
            statefile = self._checkpoint_path(name)
            if "debug" in cdict and cdict["debug"]:
                statefile = None
            if self.compilation_status(name) == PROJECT_STATUS_COMPLETE:
                continue

            logging.output_file = os.path.splitext(self.projfile)[0] + "-{}".format(name)
            logging.logprint("Starting compilation of {}".format(name))
            with threadpool_limits(limits=blas_threads, user_api='blas'):
                result, structure, vector = compiler.compile(U, depth=None, statefile=statefile)
            logging.logprint("Finished compilation of {}".format(name))
            cdict["result"] = result
            cdict["structure"] = structure
            cdict["vector"] = vector
            self._compilations[name] = (U, cdict)
            self._save()
            logging.logprint("Recorded results from compilation.")

            checkpoint.delete(statefile)
            logging.logprint("Deleted checkpoint file.")
            self.status()

    def complete(self):
        for name in self._compilations:
            s = self.compilation_status(name)
            if s == PROJECT_STATUS_PROGRESS or s == PROJECT_STATUS_NOTBEGUN:
                return False
        return True

    def status(self):
        for name in self._compilations:
            s = self.compilation_status(name)
            msg = ""
            if s == PROJECT_STATUS_COMPLETE:
                msg = "Complete!"
            elif s == PROJECT_STATUS_PROGRESS:
                msg = "In progress..."
            elif s == PROJECT_STATUS_NOTBEGUN:
                msg = "Not started."
            elif s == PROJECT_STATUS_DEBUGING:
                msg = "Debug."

            print("{}\t{}".format(name,msg))

    def compilation_status(self, name):
        _, cdict = self._compilations[name]
        if "debug" in cdict and cdict["debug"]:
            return PROJECT_STATUS_DEBUGING
        elif os.path.exists(self._checkpoint_path(name)):
            return PROJECT_STATUS_PROGRESS
        elif "structure" in cdict and "vector" in cdict:
            return PROJECT_STATUS_COMPLETE
        else:
            return PROJECT_STATUS_NOTBEGUN

    def get_result(self, name):
        _, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
            return None, None

        return cdict["structure"], cdict["vector"]

    def verify_result(self, name, count=1000):
        original, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("The compilation {} has not been completed.  Please run the project to finish the compilation.")
            return

        final = cdict["result"]
        maxs, total, mins = sc.utils.random_vector_evaluation(original, final, count)

        print("Max: {}%\nAverage: {}%\nMin: {}%\n".format(maxs*100.0, total*100.0, mins*100.0))


    def assemble(self, name, language=assembler.ASSEMBLY_OPENQASM, write_location=None):
        _, cdict = self._compilations[name]
        if not "structure" in cdict or not "vector" in cdict:
            print("this compilation has not been completed.  please run the project to complete the compilation.")
            return None, None

        out = assembler.assemble(cdict["structure"], cdict["vector"], language, write_location)
        if write_location == None:
            print(out)

