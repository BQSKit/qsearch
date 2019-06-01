import pickle
import os

def save(state, filepath=None):
    if filepath is None:
        return
    with open(filepath, "wb") as tmpfile:
        pickle.dump(state, tmpfile, pickle.HIGHEST_PROTOCOL)

def recover(filepath=None):
    if filepath is None:
        return
    try:
        with open(filepath, "rb") as tmpfile:

            return pickle.load(tmpfile)

    except Exception:
        delete(filepath)
        return None

def delete(filepath):
    if filepath is None:
        return
    try:
        os.remove(filepath)
    except Exception:
        pass

