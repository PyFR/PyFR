import functools as ft
import itertools as it
import os
from pathlib import Path
import pickle
import random
import uuid

from platformdirs import user_cache_dir


def memoize(origfn=None, maxsize=None):
    def memoizefn(meth):
        cattr = f'_memoize_cache@{meth.__name__}@{id(meth)}'

        @ft.wraps(meth)
        def newmeth(self, *args, **kwargs):
            try:
                cache = getattr(self, cattr)
            except AttributeError:
                setattr(self, cattr, cache := {})

            if kwargs:
                key = (args, tuple(kwargs.items()))
            else:
                key = args

            try:
                return cache[key]
            except KeyError:
                pass
            except TypeError:
                key = pickle.dumps((args, kwargs))

                try:
                    return cache[key]
                except KeyError:
                    pass

            if maxsize and len(cache) == maxsize:
                rmk = next(it.islice(cache, random.randrange(maxsize), None))
                del cache[rmk]

            res = cache[key] = meth(self, *args, **kwargs)
            return res

        return newmeth

    return memoizefn(origfn) if origfn else memoizefn


class ObjectCache:
    def __init__(self, suffix, *, maxsize=128*1024**2):
        self.suffix = suffix

        # See if the cache is enabled
        self.enabled = f'PYFR_{suffix.upper()}_DISABLE_CACHE' not in os.environ

        # Determine the cache directory
        cdir = os.environ.get(f'PYFR_{suffix.upper()}_CACHE_DIR')
        if not cdir:
            cdir = Path(user_cache_dir('pyfr')) / suffix

        self.cachedir = Path(cdir).absolute()

        if self.enabled:
            self.cachedir.mkdir(parents=True, exist_ok=True)
            self._prune_cache(maxsize)

    def get_path(self, k):
        return self.cachedir / k if self.enabled else None

    def get_bytes(self, k):
        try:
            return self.get_path(k).read_bytes()
        except (AttributeError, IOError):
            return None

    def set_with_bytes(self, k, bytes):
        try:
            if self.enabled:
                ctpath = self.cachedir / str(uuid.uuid4())

                ctpath.write_bytes(bytes)
                return ctpath.rename(self.get_path(k))
        except OSError:
            pass

        return None

    def set_with_path(self, k, fpath):
        try:
            if self.enabled:
                fpath = Path(fpath).rename(self.cachedir / str(uuid.uuid4()))
                return fpath.rename(self.get_path(k))
        except OSError:
                pass

        return None

    def _prune_cache(self, maxsize):
        files = {f: f.stat() for f in self.cachedir.iterdir() if f.is_file()}
        csize = sum(fs.st_size for fs in files.values())

        if csize > maxsize:
            for f, fs in sorted(files.items(), key=lambda f: f[1].st_atime):
                f.unlink(missing_ok=True)
                csize -= fs.st_size

                if csize <= maxsize:
                    break
