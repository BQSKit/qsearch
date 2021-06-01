"""
A class for holding and managing options that are passed to various other classes in the Qsearch suite.

Options objects work like dictionaries, but if an Options object is queried for an item and it does not have it, it first checks its defaults and smart_defaults properties for the item before throwing an error.  This allows the setting of default values which are easily overridden by user-provided values.  The smart_defaults dictionary contains functions that return an object, allowing for default behavior that is dependent on other settings within the Options object.

Options objects are also designed to be easily combinable through functions such as update and updated.

Options objects are used ubiquitously throughout Qsearch
"""

import pickle
import os

_options_actual_parameters = ["defaults", "smart_defaults", "required", "cache", "load_error"]

class Options():
    """This class manages options that are passed between various Qsearch objects."""
    def __init__(self, defaults={}, smart_defaults={}, **xtraargs):
        self.defaults = dict()
        self.smart_defaults = dict()
        self.required = set()
        self.cache = dict()
        self.load_error = None
        self.update(**xtraargs)
        self.set_defaults(**defaults)
        self.set_smart_defaults(**smart_defaults)
        


    def filtered(self, *names):
        """Returns an Options object with only parameters in the specified list names."""
        new_dict = {name:self.__dict__[name] for name in names if name in self.__dict__ and not name in _options_actual_parameters}
        new_defaults = {name:self.defaults[name] for name in names if name in self.defaults}
        new_smart_defaults = {name:self.smart_defaults[name] for name in names if name in self.smart_defaults}
        
        newOptions = Options(**new_defaults)
        newOptions.__dict__.update(new_dict)
        newOptions.smart_defaults = new_smart_defaults
        return newOptions

    def __getitem__(self, name):
        return getattr(self, name)

    def __delitem__(self, name):
        if name in _options_actual_parameters:
            raise AttributeError("This is one of the essential parameters and cannot be removed")
        del self.__dict__[name]

    def __getattr__(self, name):
        if name in _options_actual_parameters:
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
            try:
                retval = smartfunc(self)
                self.cache[name] = retval
            finally:
                self.required.remove(name)
            return retval
        elif name in self.defaults:
            return self.defaults[name]
        raise AttributeError("Could not find option '{}'".format(name))

    def __setattr__(self, name, value):
        if name not in _options_actual_parameters:
            self.cache = dict()
        super().__setattr__(name, value)

    def __contains__(self, name):
        if name in self.__dict__:
            return True
        elif name in self.required:
            return False
        elif name in self.defaults:
            return True
        elif name in self.smart_defaults:
            return True
        else:
            return False

    def manually_entered(self, *names, location="dict", operator="all"):
        if location == "dict":
            for name in names:
                if name in self.__dict__:
                    if operator == "any":
                        return True
                elif operator == "all":
                    return False
        elif location == "defaults":
            for name in names:
                if name in self.defaults:
                    if operator == "any":
                        return True
                elif operator == "all":
                    return False
        elif location == "smart_defaults":
            for name in names:
                if name in self.smart_defaults:
                    if operator == "any":
                        return True
                elif operator == "all":
                    return False

    def empty_copy(self):
        """Create an Options object with the same defaults but without any specific values."""
        newOptions = Options(self.defaults)
        newOptions.smart_defaults = self.smart_defaults
        newOptions.required = self.required.copy()
        return newOptions

    def copy(self):
        """Create a full copy of an Options object."""
        newOptions = Options(self.defaults)
        newOptions.smart_defaults.update(self.smart_defaults)
        newOptions._update_dict(self.__dict__)
        newOptions.required = self.required.copy()
        newOptions.cache = self.cache.copy()
        return newOptions

    def __copy__(self):
        newOptions = Options(self.defaults)
        newOptions.smart_defaults.update(self.smart_defaults)
        newOptions._update_dict(self.__dict__)
        newOptions.required = self.required.copy()
        newOptions.cache = self.cache.copy()
        return newOptions

    def updated(self, other=None, **xtraargs):
        """Return a new Options object that is a copy of this object, updated with the contents of other and xtraargs."""
        newOptions = self.copy()
        if other is not None:
            newOptions._update_dict(other.__dict__)
            newOptions.set_defaults(**other.defaults)
            newOptions.set_smart_defaults(**other.smart_defaults)
            newOptions.required = self.required.union(other.required)
        newOptions._update_dict(xtraargs)
        newOptions.cache = dict()
        return newOptions

    def update(self, other=None, **xtraargs):
        """Mutate the current Options object with the contents of other and xtraargs."""
        if other is not None:
            self._update_dict(other.__dict__)
            self.set_defaults(**other.defaults)
            self.set_smart_defaults(**other.smart_defaults)
            self.required.update(other.required)
        self._update_dict(xtraargs)
        self.cache = dict()


    def _update_dict(self, otherdict):
        for name in otherdict:
            if name in _options_actual_parameters:
                continue
            self.__dict__[name] = otherdict[name]

    def set_defaults(self, **args):
        """
        Set default values for this Options object.
        
        If an Options object is queried for a value, and it does not contain it, it will check its defaults list before throwing an error.
        """
        self.defaults.update(args)
        for name in args:
            if name in self.smart_defaults:
                del self.smart_defaults[name]
        self.cache = dict()

    # if a smart_default function is specified, it will be called instead of falling back to normal defaults
    # smart default functions should take an Options class and return a value
    def set_smart_defaults(self, **args):
        """
        Set smart_defaults values for this Options object.

        If an Options object is queried for a value, and it does not contain it, it will check its smart_defaults list before throwing an error.  If it does find a function in smart_defaults, it calls that function, passing itself as the argument, and returns the return value of that function, caching it for next time.
        """
        self.smart_defaults.update(args)
        for name in args:
            if name in self.defaults:
                del self.defaults[name]
        self.cache = dict()

    def make_required(self, *names):
        """Marking names as required will cause the Options object to throw an error if it does not contain it, even if it has defaults or smart_defaults defined."""
        self.required.update(names)
        self.cache = dict()

    def remove_defaults(self, *names):
        """Removes the defaults for the specified names."""
        for name in names:
            if name in self.defaults:
                del self.defaults[name]
        self.cache = dict()

    def remove_smart_defaults(self, *names):
        """Removes the smart_defaults for the specified names."""
        for name in names:
            if name in self.smart_defaults:
                del self.smart_defaults[name]
        self.cache = dict()

    def generate_cache(self):
        """Caches valuesa for all functions in smart_defaults."""
        for key in self.smart_defaults:
            try:
                getattr(self, key) 
            except:
                pass
            # this may look weird, but calling getattr on these keys will populate
            # the cache with any keys that have not been already cached.

    def save(self, filepath=None):
        """Saves the Options object to a file, or to a returned tuple."""
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

        if filepath is None:
            return (main_dict, defaults_dict, smart_defaults_dict)
        pickle.dump((main_dict, defaults_dict, smart_defaults_dict), filepath)


    def load(self, filepath_or_tuple, strict=False):
        """
        Loads the Options object from a file or tuple.
        
        If strict is left as False, the Options object will attempt to gracefully handle errors when loading its contents, relying on its ability to fallback to defaults or smart_defaults if those are able to load successfully.

        If strict is set to True, the Options object will throw an error upon any error while loading.
        """
        try:
            if type(filepath_or_tuple) is tuple:
                main_dict, defaults_dict, smart_defaults_dict = filepath_or_tuple
            else:
                main_dict, defaults_dict, smart_defaults_dict = pickle.load(filepath)
        except Exception as e:
            if strict:
                raise
            else:
                self.load_error = e
            return

        for name in main_dict:
            try:
                self.__dict__[name] = pickle.loads(main_dict[name])
            except Exception as e:
                if strict:
                    raise
                else:
                    self.load_error = e

        for name in defaults_dict:
            try:
                self.defaults[name] = pickle.loads(defaults_dict[name])
            except Exception as e:
                if strict:
                    raise
                else:
                    self.load_error = e

        for name in smart_defaults_dict:
            try:
                self.smart_defaults[name] = pickle.loads(smart_defaults_dict[name])
            except Exception as e:
                if strict:
                    raise
                else:
                    self.load_error = e

    def __getstate__(self):
        return self.save()

    def __setstate__(self, state):
        self.defaults = dict()
        self.smart_defaults = dict()
        self.required = set()
        self.cache = dict()
        self.load_error = None

        try:
            self.load(state, strict=True)
        except Exception as e:
            self.load(state, strict=False)
            self.load_error = e

