from dataclasses import dataclass


@dataclass(frozen=True)
class FieldInfo:
    name: str

    kind: str

    dtype: object = None
    source: str = 'unknown'

    components: tuple = ()

    data_index: int = None

    @property
    def ncomps(self):
        return len(self.components)
