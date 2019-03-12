import functools

"""
Decorator to create lazy functions
"""


class lazy(object):
    """
    lazy descriptor
    Used as a decorator to create lazy functions. These
    are evaluated on first use.
    """

    def __init__(self, func):
        self.__func = func
        functools.wraps(self.__func)(self)
        self.value = None
        self.is_computed = False

    def __get__(self, instance, owner):
        if instance is None:
            return self

        if not hasattr(instance, '__dict__'):
            raise AttributeError("'{}' object has no attribute '__dict__'".format(owner.__name__,))

        name = self.__name__
        if name.startswith('__') and not name.endswith('__'):
            name = '_lazy_cache_{}{}'.format(owner.__name__, name)
        else:
            name = '_lazy_cache_{}'.format(name)

        if name not in instance.__dict__:
            value = self.__func(instance)
            instance.__dict__[name] = value

        def emit():
            return instance.__dict__[name]

        return emit
