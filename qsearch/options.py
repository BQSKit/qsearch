"""
A class for holding and managing options that are passed to various other classes in the Qsearch suite.
"""

import pickle
import os

_options_actual_paramters = ["defaults", "smart_defaults", "required", "cache"]

class Options():
    def __init__(self, **defaults):
        self.defaults = dict()
        self.defaults.update(defaults)
        self.smart_defaults = dict()
        self.required = set()
        self.cache = dict()

    def filtered(self, *names):
        new_dict = {name:self.__dict__[name] for name in names if name in self.__dict__ and not name in _options_actual_paramters}
        new_defaults = {name:self.defaults[name] for name in names if name in self.defaults}
        new_smart_defaults = {name:self.smart_defaults[name] for name in names if name in self.smart_defaults}
        
        newOptions = Options(**new_defaults)
        newOptions.__dict__.update(new_dict)
        newOptions.smart_defaults = new_smart_defaults
        return newOptions

    def __getitem__(self, name):
        return getattr(self, name)

    def __delitem__(self, name):
        if name in _options_actual_paramters:
            raise AttributeError("This is one of the essential parameters and cannot be removed")
        del self.__dict__[name]

    def __getattr__(self, name):
        if name in _options_actual_paramters:
            raise AttributeError("This options class is corrupt as it is missing its critical members")
        elif name in self.required:
            raise AttributeError("Missing required parameter {}".format(name))
        elif name in self.cache:
            return self.cache[name]
        elif name in self.smart_defaults:
            # this behavior of removing the smart default function and putting it back is intended
            # to generate an error rather than recurse infinitely when there are interdependent smart default functions
            smartfunc = self.smart_defaults[name]
            self.required.add(name)
            retval = smartfunc(self)
            self.smart_defaults[name] = smartfunc
            self.cache[name] = retval
            self.required.remove(name)
            return retval
        elif name in self.defaults:
            return self.defaults[name]
        raise AttributeError("Could not find option {}".format(name))

    def __setattr__(self, name, value):
        if name not in _options_actual_paramters:
            self.cache = dict()
        super().__setattr__(name, value)

    def __contains__(self, name):
        if name in self.__dict__:
            return True
        elif name in self.defaults:
            return True
        elif name in self.smart_defaults:
            return True
        else:
            return False

    # creates an Options object with the same deults but without any specific values
    def empty_copy(self):
        newOptions = Options(**self.defaults)
        newOptions.smart_defaults = self.smart_defaults
        newOptions.required = self.required.copy()
        return newOptions

    def copy(self):
        newOptions = Options(**self.defaults)
        newOptions.smart_defaults.update(self.smart_defaults)
        newOptions._update_dict(self.__dict__)
        newOptions.required = self.required.copy()
        newOptions.cache = self.cache.copy()
        return newOptions

    def __copy__(self):
        newOptions = Options(**self.defaults)
        newOptions.smart_defaults.update(self.smart_defaults)
        newOptions._update_dict(self.__dict__)
        newOptions.required = self.required.copy()
        newOptions.cache = self.cache.copy()
        return newOptions

    def updated(self, other=None, **xtraargs):
        newOptions = self.copy()
        if other is not None:
            newOptions._update_dict(other.__dict__)
            newOptions.defaults.update(other.defaults)
            newOptions.smart_defaults.update(other.smart_defaults)
            newOptions.required = self.required.union(other.required)
        newOptions._update_dict(xtraargs)
        return newOptions

    def update(self, other=None, **xtraargs):
        if other is not None:
            self._update_dict(other.__dict__)
            self.defaults.update(other.defaults)
            self.smart_defaults.update(other.smart_defaults)
            self.required.update(other.required)
        self._update_dict(xtraargs)
        self.cache = dict()


    def _update_dict(self, otherdict):
        for name in otherdict:
            if name in _options_actual_paramters:
                continue
            self.__dict__[name] = otherdict[name]

    # the options class will fallback to default values if a normal value is not specified
    def set_defaults(self, **args):
        self.defaults.update(args)
        for name in args:
            if name in self.smart_defaults:
                del self.smart_defaults[name]
        self.cache = dict()

    # if a smart_default function is specified, it will be called instead of falling back to normal defaults
    # smart default functions should take an Options class and return a value
    def set_smart_defaults(self, **args):
        self.smart_defaults.update(args)
        for name in args:
            if name in self.defaults:
                del self.defaults[name]
        self.cache = dict()

    def make_required(self, *names):
        self.required.update(names)
        self.cache = dict()

    def remove_defaults(self, *names):
        for name in names:
            if name in self.defaults:
                del self.defaults[name]
        self.cache = dict()

    def remove_smart_defaults(self, *names):
        for name in names:
            if name in self.smart_defaults:
                del self.smart_defaults[name]
        self.cache = dict()

    def generate_cache(self):
        for key in self.smart_defaults:
            getattr(self, key) 
            # this may look weird, but calling getattr on these keys will populate
            # the cache with any keys that have not been already cached.

    def save(self, filepath):
        main_dict = dict()
        for name in self.__dict__:
            if not name in _options_actual_parameters:
                main_dict[name] = pickle.dumps(self.__dict__[name])

        defaults_dict = dict()
        for name in self.defaults:
            if not name in _options_actual_parameters:
                defaults_dict[name] = pickle.dumps(self.defaults[name])

        smart_defaults_dict = dict()
        for name in self.smart_defaults:
            if not name in _options_actual_parameters:
                smart_defaults_dict[name] = pickle.dumps(self.smart_defaults[name])

        pickle.dump((main_dict, defaults_dict, smart_defaults_dict), filepath)


    def load(self, filepath, strict=False):
        if not strict:
            try:
                main_dict, defaults_dict, smart_defaults_dict = pickle.load(filepath)
            except:
                return

            for name in main_dict:
                try:
                    self.__dict__[name] = pickle.loads(main_dict[name])
                except:
                    pass

            for name in defaults_dict:
                try:
                    self.defaults[name] = pickle.loads(defaults_dict[name])
                except:
                    pass

            for name in smart_defaults_dict:
                try:
                    self.smart_defaults[name] = pickle.loads(smart_defaults_dict[name])
                except:
                    pass
        else:
            main_dict, defaults_dict, smart_defaults_dict = pickle.load(filepath)

            for name in main_dict:
                self.__dict__[name] = pickle.loads(main_dict[name])

            for name in defaults_dict:
                self.defaults[name] = pickle.loads(defaults_dict[name])

            for name in smart_defaults_dict:
                self.smart_defaults[name] = pickle.loads(smart_defaults_dict[name])


