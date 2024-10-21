from collections.abc import Sequence
from typing import Generic, TypeVar, overload

import numpy as np

from ._types import TimestampsArray

IndexerValueType = TypeVar("IndexerValueType")
IndexerKeyType = int | float


class Indexer(Generic[IndexerValueType]):
    def __init__(
        self,
        keys: TimestampsArray,
        values: Sequence[IndexerValueType],
    ):
        self.values = values
        self.keys = np.array(keys)

    @overload
    def __getitem__(self, key: IndexerKeyType) -> IndexerValueType: ...

    @overload
    def __getitem__(self, key: slice) -> list[IndexerValueType]: ...

    def __getitem__(self, key: IndexerKeyType | slice) -> IndexerValueType | Sequence[IndexerValueType]:
        if isinstance(key, int | float):
            index = np.searchsorted(self.keys, [key])
            if self.keys[index] != key:
                raise IndexError()
            return self.values[int(index)]
        elif isinstance(key, slice):
            start_index, stop_index = np.searchsorted(self.keys, [key.start, key.stop])
            return self.values[start_index:stop_index]
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")
