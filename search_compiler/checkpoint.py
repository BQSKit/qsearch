import pickle
import os

tmp = "./compiler_checkpoint.obj"

def save(state):
    with open(tmp, "wb") as tmpfile:
        pickle.dump(state, tmpfile, pickle.HIGHEST_PROTOCOL)

def recover():
    try:
        with open(tmp, "rb") as tmpfile:

            return pickle.load(tmpfile)

    except Exception:
        delete()
        return None

def delete():
    try:
        os.remove(tmp)
    except Exception:
        pass

