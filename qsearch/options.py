"""
A class for holding and managing options that are passed to various other classes in the Qsearch suite.
"""


class Options():
    def __init__(self, **defaults):
        self.defaults = {}
        self.defaults.update(defaults)
        self.smart_defaults = {}
        self.required = Set()

    def filtered(self, *names):
        new_dict = {name:self.dict[name] for name in names if name in self.dict}
        new_defaults = {name:self.defaults[name] for name in names if name in self.defaults}
        new_smart_defaults = {name:self.smart_defaults[name] for name in names if name in self.smart_defaults}
        
        newOptions = Options(**new_dict)
        newOptions.defaults = new_defaults
        newOptions.smart_defaults = new_smart_defaults
        return newOptions

    def __getattr__(self, name):
        if name in required:
            raise AttributeError("Missing required parameter {}".format(name))
        elif name in self.smart_defaults:
            return self.smart_defaults[name](self)
        elif name in self.defaults:
            return self.defaults[name]
        raise AttributeError()

    # creates an Options object with the same deults but without any specific values
    def empty_copy(self):
        newOptions = Options(**self.defaults)
        newOptions.smart_defaults = self.smart_defaults
        newOptions.required = self.required.copy()
        return newOptions

    def copy(self):
        newOptions = Options(**self.defaults)
        newOptions.smart_defaults.update(self.smart_defaults)
        newOptions.__dict__.update(self.__dict__)
        newOptions.required = self.required.copy()
        return newOptions

    def __copy__(self):
        newOptions = Options(**self.defaults)
        newOptions.smart_defaults.update(self.smart_defaults)
        newOptions.__dict__.update(self.__dict__)
        newOptions.required = self.required.copy()
        return newOptions

    def updated(self, other):
        newOptions = self.copy()
        newOptions.__dict__.update(other.__dict__)
        newOptions.defaults.update(other.defaults)
        newOptions.smart_defaults.update(other.smart_defaults)
        newOptions.required = self.required.union(other.required)
        return newOptions

    def update(self, other=None, **xtraargs):
        if other is not None:
            self.__dict__.update(other.__dict__)
            self.defaults.update(other.defaults)
            self.smart_defaults.update(other.smart_defaults)
            self.required.update(other.required)
        else:
            self.__dict__.update(xtraargs)


    # the options class will fallback to default values if a normal value is not specified
    def set_defaults(self, **args):
        self.defaults.update(args)

    # if a smart_default function is specified, it will be called instead of falling back to normal defaults
    # smart default functions should take an Options class and return a value
    def set_smart_defaults(self, **args):
        self.smart_defaults.update(args)

    def make_required(*names):
        self.required.update(names)

