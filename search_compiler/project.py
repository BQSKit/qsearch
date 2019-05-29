import os
import shutil
import pickle
from .compiler import SearchCompiler
from . import gatesets
from . import logging, checkpoint, utils

class Project:
    def __init__(self, directory, debug=False):
        self.directory = directory
        self._projpath = os.path.join(directory, ".search_compiler_projfile")
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            with open(self._projpath, "rb") as projfile:
                self._compilations, self.compiler_config = pickle.load(projfile)
        except Exception:
            self._compilations = dict()
            self.compiler_config = dict()

    def _save(self):
        with open(self._projpath, "wb") as projfile:
            pickle.dump((self._compilations, self.compiler_config), projfile)


    def add_compilation(self, name, U, debug=False, handle_existing=None):
        if name in self._compilations:
            if handle_existing == "ignore":
                return
            elif handle_existing == "overwrite":
                self.remove_compilation(name)
            else:
                raise Exception("A compilation with name {} already exists.  To change it, remove it and then add it again.".format(name))
                return

        self._compilations[name] = (U, {"debug" : debug})
        folder = os.path.join(self.directory, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self._save()

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
        # at some point allow the compiler to be configureable via the compiler_config dictionary
        threshold = 1e-10
        gateset = gatesets.QubitCNOTLinear()
        error_func = utils.astar_heuristic
        compiler = SearchCompiler(threshold=threshold, gateset=gateset, error_func=error_func)
        for name in self._compilations:
            U, params = self._compilations[name]
            folder = os.path.join(self.directory, name)
            if "debug" in params and params["debug"]:
                shutil.rmtree(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            if project_complete(folder, name):
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
            _, _, params = self._compilations[name]
            if "debug" in params and params["debug"]:
                continue
            if not project_complete(os.path.join(self.directory, name), name):
                return False
        return True

    def status(self):
        for name in self._compilations:
            _, params = self._compilations[name]
            msg = ""
            if project_complete(os.path.join(self.directory, name), name):
                msg = "Complete!"
            elif os.path.exists(os.path.join(self.directory, name, ".checkpoint")):
                msg = "In progress..."
            else:
                msg = "Not started."
            if "debug" in params and params["debug"]:
                msg = "Debug."

            print("{}\t{}".format(name,msg))


def project_complete(folder, name):
    return not os.path.exists(os.path.join(folder, ".checkpoint")) and os.path.exists(os.path.join(folder, "{}-structure.txt".format(name))) and os.path.exists(os.path.join(folder, "{}-vector.txt".format(name)))

