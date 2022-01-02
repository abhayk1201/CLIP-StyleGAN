import atexit
import inspect
import shelve
from pathlib import Path

import h5py
import numpy as np

cache_path = Path(".cache")
cache_path.mkdir(exist_ok=True, parents=True)


class Memoize(object):
    """
    Memoize a function via shelve storage (on disk).
    >>> @Memoize
    >>> def costly_func(*args, **kwargs):
    >>>     # do some costly stuff
    >>>     return costly_stuff
    """

    def __init__(self, func):
        self.func = func
        self.cache = shelve.open(str(cache_path / func.__name__), "c")
        atexit.register(self.cache.close)

    def __call__(self, *args, **kwargs):
        key = self.key(args, kwargs)
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)
        return self.cache[key]

    def normalize_args(self, args, kwargs):
        spec = inspect.getargs(self.func.__code__).args
        return {**kwargs, **dict(zip(spec, args))}

    def key(self, args, kwargs):
        a = self.normalize_args(args, kwargs)
        return str(sorted(a.items()))


class MemoizeNumpy(Memoize):
    """
    Memoize a function which produces a single numpy output.
    >>> @MemoizeNumpy
    >>> def costly_func(*args, **kwargs):
    >>>     # do some costly stuff
    >>>     return costly_stuff
    """

    def __init__(self, func):
        self.func = func
        self.cache = h5py.File(f"{cache_path}/{func.__name__}.hdf5", "a")
        atexit.register(self.cache.close)

    def __call__(self, *args, **kwargs):
        key = self.key(args, kwargs)
        if key not in self.cache:
            output = self.func(*args, **kwargs)

            assert isinstance(output, np.ndarray), "Output must be a single numpy array"

            self.cache.create_dataset(key, data=output, compression="lzf")
        else:
            dset = self.cache.get(key)

            # Create empty numpy array
            output = np.zeros(shape=dset.shape, dtype=dset.dtype)
            dset.read_direct(output)

        return output
