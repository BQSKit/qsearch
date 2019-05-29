import os
import shutil
import pickle
from .compiler import SearchCompiler
from .solver import CMA_Solver
from . import logging, checkpoint, utils, gatesets

PROJECT_STATUS_PROGRESS = 1
PROJECT_STATUS_COMPLETE = 2
PROJECT_STATUS_NOTBEGUN = 3
PROJECT_STATUS_DEBUGING = 4

class Project:
    def __init__(self, directory, debug=False):
        self.directory = directory
        self._projpath = os.path.join(directory, ".search_compiler_projfile")
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            with open(self._projpath, "rb") as projfile:
                self._compilations, self._compiler_config = pickle.load(projfile)
        except Exception:
            self._compilations = dict()
            self.compiler_config = dict()

    def _save(self):
        with open(self._projpath, "wb") as projfile:
            pickle.dump((self._compilations, self._compiler_config), projfile)
    
    def _config(self, keyword, default):
        if keyword in self._compiler_config:
            return self._compiler_config[keyword]
        else:
            return default


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
        folder = os.path.join(self.directory, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self._save()

    def configure_compiler(self, keyword, value, force=False):
        if not force:
            for name in self._compilations:
                s = self.compilation_status(name)
                if s == PROJECT_STATUS_COMPLETE or s == PROJECT_STATUS_PROGRESS:
                    warn("This project contains compilations which have been completed or have been started.  Changing the compiler configuration will delete the existing progress.  Call configure_compiler with the 'force' option set to True to reset these compilations and reconfigure the compiler.", RuntimeWarning, stacklevel=2)
                    self.status()
                    return
        for name in self._compilations:
            folder = os.path.join(self.directory, name)
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        
        self._compiler_config[keyword] = value



    def remove_compilation(self, name):
        shutil.rmtree(os.path.join(self.directory, name))
        self._compilations.pop(name)
        self._save()

    def clear(self):
        shutil.rmtree(self.directory)
        os.makedirs(self.directory)
        self._compilations = dict()
        self.compiler_config = dict()
        self._save()


    def run(self):
        threshold = self._config("threshold", 1e-10)
        gateset = self._config("gateset", gatesets.QubitCNOTLinear())
        error_func = self._config("error_func", utils.astar_heuristic)
        solver = self._config("solver", CMA_Solver())
        compiler = SearchCompiler(threshold=threshold, gateset=gateset, error_func=error_func, solver=solver)
        self.status()
        for name in self._compilations:
            U, params = self._compilations[name]
            folder = os.path.join(self.directory, name)
            if "debug" in params and params["debug"]:
                shutil.rmtree(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            if self.compilation_status(name) == PROJECT_STATUS_COMPLETE:
                continue

            logging.output_file = os.path.join(folder, "{}-log.txt".format(name))
            logging.logprint("Starting compilation of {}".format(name))
            result, structure, vector = compiler.compile(U, depth=None, statefile=os.path.join(folder, ".checkpoint"))
            logging.logprint("Finished compilation of {}".format(name))
            with open(os.path.join(folder, "{}-target.txt".format(name)), "w") as target:
                target.write(repr(U))
                logging.logprint("recorded target")
            with open(os.path.join(folder, "{}-final.txt".format(name)), "w") as target:
                target.write(repr(result))
                logging.logprint("recorded result")
            with open(os.path.join(folder, "{}-structure.txt".format(name)), "w") as target:
                target.write(repr(structure))
                logging.logprint("recorded structure")
            with open(os.path.join(folder, "{}-vector.txt".format(name)), "w") as target:
                target.write(repr(vector))
                logging.logprint("recorded vector")

            checkpoint.delete(os.path.join(folder, ".checkpoint"))
            logging.logprint("deleted checkpoint file")
            self.status()




    def complete(self):
        for name, in self._compilations:
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
        folder = os.path.join(self.directory, name)
        _, params = self._compilations[name]
        if "debug" in params and params["debug"]:
            return PROJECT_STATUS_DEBUGING
        elif os.path.exists(os.path.join(folder, ".checkpoint")):
            return PROJECT_STATUS_PROGRESS
        elif os.path.exists(os.path.join(folder, "{}-structure.txt".format(name))) and os.path.exists(os.path.join(folder, "{}-vector.txt".format(name))):
            return PROJECT_STATUS_COMPLETE
        else:
            return PROJECT_STATUS_NOTBEGUN

