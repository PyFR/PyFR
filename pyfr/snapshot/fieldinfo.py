from dataclasses import dataclass, field as dc_field


@dataclass(frozen=True)
class FieldInfo:
    name: str

    kind: str
    ncomps: int

    dtype: object = None
    source: str = 'unknown'

    components: tuple = dc_field(default_factory=tuple)

    data_index: int = None
