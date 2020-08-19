import pickle
import os
from . import options


class Checkpoint():
    def __init__(self, opt=options.Options(), **xtraargs):
        self.options = opt.updated(**xtraargs)

    def save(self, state):
        # should save the passed state
        raise NotImplementedError

    def recover(self):
        # should return the previously saved state
        raise NotImplementedError

    def delete(self):
        # should delete all the checkpoint data
        raise NotImplementedError


class FileCheckpoint(Checkpoint):
    def save(self, state):
        if self.options.statefile == None:
            return
        with open(self.options.statefile, "wb") as tmpfile:
            pickle.dump(state, tmpfile, pickle.HIGHEST_PROTOCOL)

    def recover(self):
        if self.options.statefile == None:
            return None
        try: 
            with open(self.options.statefile, "rb") as tmpfile:
                return pickle.load(tmpfile)
        except:
            return None

    def delete(self):
        if self.options.statefile == None:
            return
        try:
            os.remove(self.options.statefile)
        except:
            return

def ChildCheckpoint(Checkpoint):
    def __init__(self, parent, opt=options.Options(), **xtraargs):
        self.parent = parent
        super().__init__(opt, **xtraargs)

    def save(self, state):
        self.parent.save((self.parentdata, state))

    def recover(self):
        recdata = self.parent.recover()
        if recdata is None:
            self.parentdata = None
            return None
        else:
            self.parentdata = recdata[0]
            return recdata[1]

    def delete(self):
        self.parent.delete()

