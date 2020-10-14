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

class ChildCheckpoint(Checkpoint):
    def __init__(self, parent, opt=options.Options(), **xtraargs):
        super().__init__(opt, **xtraargs)
        self.parent = parent
        self.recover()

    def save(self, state):
        self.state = state
        self.parent.save((self.parentstate, self.state))

    def save_parent(self, parentstate):
        self.parentstate = parentstate
        self.save(self.state)

    def recover(self):
        recdata = self.parent.recover()
        if recdata is None:
            self.parentstate = None
            self.state = None
            return None
        else:
            self.parentstate, self.state = recdata
            return self.state

    def recover_parent(self):
        self.recover()
        return self.parentstate

    def delete(self):
        self.parent.delete()

