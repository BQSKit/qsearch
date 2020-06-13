"""
A class for holding and managing options that are passed to various other classes in the Qsearch suite.
"""


class Options():
    def __init__(self, **defaults):
        self.defaults = defaults
        self.smart_defaults = {}

    def filtered(self, *names):
        new_dict = {name:self.dict[name] for name in names if name in self.dict}
        new_defaults = {name:self.defaults[name] for name in names if name in self.defaults}
        new_smart_defaults = {name:self.smart_defaults[name] for name in names if name in self.smart_defaults}
        
        newOptions = Options(**new_dict)
        newOptions.defaults = new_defaults
        newOptions.smart_defaults = new_smart_defaults
        return newOptions

    def __getattr__(self, name):
        if name in self.smart_defaults:
            return self.smart_defaults[name](self)
        elif name in self.defaults:
            return self.defaults[name]
        raise AttributeError()

    # creates an Options object with the same defaults but without any specific values
    def empty_copy():
        newOptions = Options(**self.defaults)
        newOptions.smart_defaults = self.smart_defaults

    # the options class will fallback to default values if a normal value is not specified
    def set_defaults(self, **args):
        for key in args:
            self.defaults[key] = args[key]

    # if a smart_default function is specified, it will be called instead of falling back to normal defaults
    # smart default functions should take an Options class and return a value
    def set_smart_defaults(self, **args):
        for key in args:
            self.defaults[key] = args[key]

