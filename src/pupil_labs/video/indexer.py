from collections.abc import Sequence
from typing import Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt

IndexerValue = TypeVar("IndexerValueType")
IndexerKey = np.int64 | np.float64 | int | float
IndexerKeys = npt.NDArray[np.float64 | np.int64] | list[int | float]


class Indexer(Generic[IndexerValue]):
    def __init__(
        self,
        keys: IndexerKeys,
        values: Sequence[IndexerValue],
    ):
        self.values = values
        self.keys = np.array(keys)

    @overload
    def __getitem__(self, key: IndexerKey) -> IndexerValue: ...

    @overload
    def __getitem__(self, key: slice) -> list[IndexerValue]: ...

    def __getitem__(
        self, key: IndexerKey | slice
    ) -> IndexerValue | Sequence[IndexerValue]:
        if isinstance(key, int | float):
            index = np.searchsorted(self.keys, [key])[0]
            if self.keys[index] != key:
                raise IndexError()
            return self.values[int(index)]
        elif isinstance(key, slice):
            start_index, stop_index = np.searchsorted(self.keys, [key.start, key.stop])
            return self.values[start_index:stop_index]
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")
