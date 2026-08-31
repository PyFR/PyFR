class _StorageBase:
    def same_storage(self, other):
        return self.storage_root is other.storage_root

    def bind(self, root, basedata, offset):
        self.storage_root = root
        self.onalloc(basedata, offset)


class _Arena:
    def __init__(self, alignb):
        self.alignb = alignb
        self.nbytes = 0
        self._pending = []
        self._children = []
        self._sealed = False

    def _check_open(self):
        if self._sealed:
            raise RuntimeError('Extent has already been committed')

    def _rsize(self, obj):
        return obj.nbytes - obj.nbytes % -self.alignb


class _Struct(_Arena):
    def reserve(self, obj):
        self._check_open()

        self._pending.append((obj, self.nbytes))
        self.nbytes += self._rsize(obj)

    def union(self):
        self._check_open()

        u = _Union(self.alignb)
        self._children.append(u)
        return u

    def _seal(self):
        # Lay each union out at the tail of the struct
        for u in self._children:
            u._seal()
            self._pending.extend((obj, self.nbytes + off)
                                 for obj, off in u._pending)
            self.nbytes += u.nbytes
            u._pending.clear()

        self._children.clear()
        self._sealed = True


class _Union(_Arena):
    def reserve(self, obj):
        self._check_open()

        self._pending.append((obj, 0))
        self.nbytes = max(self.nbytes, self._rsize(obj))

    def struct(self):
        self._check_open()

        s = _Struct(self.alignb)
        self._children.append(s)
        return s

    def _seal(self):
        # Overlay each struct at the base of the union
        for s in self._children:
            s._seal()
            self._pending.extend(s._pending)
            self.nbytes = max(self.nbytes, s.nbytes)
            s._pending.clear()

        self._children.clear()
        self._sealed = True


class Extent(_Union, _StorageBase):
    def __init__(self, alignb):
        super().__init__(alignb)

        self.basedata = None
        self.storage_root = self

        # Route direct reservations through an implicit primary struct
        self._primary = self.struct()

    def reserve(self, obj):
        self._primary.reserve(obj)

    def union(self):
        return self._primary.union()

    def commit(self, alloc_fn):
        self._check_open()
        self._seal()

        # Check nothing has been reserved into two arenas
        objs = [obj for obj, _ in self._pending]
        if len(objs) != len(set(map(id, objs))):
            raise RuntimeError('Object reserved into multiple arenas')

        self.basedata = alloc_fn(self.nbytes)

        for obj, offset in self._pending:
            obj.bind(self, self.basedata, offset)

        self._pending.clear()
