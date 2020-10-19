"""
This module defines the Checkpoint class, which is used for storing intermediate state while compiling, to allow an interrupted compilation to resume at a later time.

Two default implementations are provided.  It is recommended that you look at FileCheckpoint as an example if you are interested in writing your own implementation.

Attributes:
    FileCheckpoint : Saves and recovers the intermediate state from a file, specified as "statefile" in the options.
    ChildCheckpoint : Allows for hierarchial checkpointing, which is useful in cases where there are sub-compilers, such as with LEAP.
"""

import pickle
import os
from . import options


class Checkpoint():
    """This class is used for storing intermediate state while compiling, to allow an interrupted compilation to resume at a later time."""
    def __init__(self, options=options.Options()):
        self.options = options

    def save(self, state):
        """
        Save the passed state to be recovered later.
        Args:
            state (object): A Python object representing the intermediate state of the compilation.  Usually a dictionary, but it could be anything.
        """
        raise NotImplementedError

    def recover(self):
        """
        Return the state previously stored with save(state).

        Returns:
            object : A Python object equivalent to the object originally stored via save(state), or None if no state is saved.
        """
        raise NotImplementedError

    def delete(self):
        """
        Delete the state that was stored such that None will be returned next time recover() is called.
        """
        raise NotImplementedError


class FileCheckpoint(Checkpoint):
    """This Checkpoint will store the state in the file specified in the options as statefile.

        Options:
            statefile : A string with a filepath where the state will be stored, or None, in which case no state will be stored and None will always be returned by recover()
    """

    def __init__(self, options=options.Options()):
        super().__init__(options)
        options.set_defaults(statefile=None)

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
    """This Checkpoint is used for hierarchial checkpointing for when there is a sub-compiler, such as in LEAP.

        Options:
            parent (required) : The Checkpoint class that the creator of the ChildCheckpoint was passed.

        Below is an explanation of how ChildCheckpoint works.  See leap_compiler for an example.

        My compiler class, ParentCompiler, is passed a FileCheckpoint as options.checkpoint.
        I create a ChildCheckpoint with the FileCheckpoint as the parent: child_checkpoint = ChildCheckpoint(Options(parent=options.checkpoint))
        I pass this ChildCheckpoint to the sub-compiler I create: sub_compiler = SubCompiler(Options(checkpoint=child_checkpoint))
        
        To the SubCompiler, the passed ChildCheckpoint will behave as any other Checkpoint would be expected to behavior, saving state with save(state), and recovering it with recover(), and deleting it with delete().

        As the ParentCompiler, you save your state with save_parent(parentstate), recover it with recover_parent(), and deleting it with delete_parent(), making these function calls to child_checkpoint instead of interacting directly with the FileCheckpoint that was originally passed via options.

        The states of both ParentCompiler and SubCompiler will get saved in a manner specified by the original FileCheckpoint.

        ChildCheckpoint fully conforms to Checkpoint, and makes no assumptions about its parent, so it is compatible with any class that makes use ofa Checkpoint and works with any Checkpoint as a parent.  This means you can even have multiple layers of nested ChildCheckpoint.

        However, the class creating the ChildCheckpoint must be sure to use the parent functions.

        Also, note that calling delete_parent() also deletes the state for the child.  However, this is rather uncommon because usually it is the creator of the Checkpoint that calls delete, not the class it is passed to.  For example, Project will call delete() to delete the checkpoint from a Compiler.  ParentCompiler might call delete() to delete the state of SubCompiler once SubCompiler has finished (in fact, this happens in leap_compiler).
    """
    def __init__(self, options=options.Options()):
        super().__init__(options)
        self.options.make_required("parent")
        self.recover()

    def save(self, state):
        self.state = state
        self.options.parent.save((self.parentstate, self.state))

    def save_parent(self, parentstate):
        """Saves the parentstate alongside the child state."""
        self.parentstate = parentstate
        self.save(self.state)

    def recover(self):
        recdata = self.options.parent.recover()
        if recdata is None:
            self.parentstate = None
            self.state = None
            return None
        else:
            self.parentstate, self.state = recdata
            return self.state

    def recover_parent(self):
        """Recovers the parentstate."""
        self.recover()
        return self.parentstate

    def delete(self):
        self.save(None)

    def delete_parent(self):
        """Deletes the state.  Note that this delete both the parentstate and the child state."""
        self.options.parent.delete()

