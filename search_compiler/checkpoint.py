import pickle
import os

tmp = ".compiler_checkpoint.obj"

def save(state, filepath=None):
    loc = filepath
    if loc is None:
        loc = tmp
    with open(loc, "wb") as tmpfile:
        pickle.dump(state, tmpfile, pickle.HIGHEST_PROTOCOL)

def recover(filepath=None):
    loc = filepath
    if loc is None:
        loc = tmp
    try:
        with open(loc, "rb") as tmpfile:

            return pickle.load(tmpfile)

    except Exception:
        delete(loc)
        return None

def delete(filepath):
    loc = filepath
    if loc is None:
        loc = tmp
    try:
        os.remove(loc)
    except Exception:
        pass

