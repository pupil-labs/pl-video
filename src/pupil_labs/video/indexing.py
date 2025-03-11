from collections.abc import Sized
from typing import Generic, TypeVar, overload

import numpy as np
import numpy.typing as npt

from pupil_labs.video.array_like import ArrayLike

IndexerValue = TypeVar("IndexerValue", covariant=True)
IndexerKey = np.uint32 | np.int32 | np.uint64 | np.int64 | np.float64 | int | float
IndexerKeys = npt.NDArray[np.float64 | np.int64] | list[int | float]


class Indexer(Generic[IndexerValue]):
    def __init__(
        self,
        keys: IndexerKeys,
        values: ArrayLike[IndexerValue],
    ):
        self.values = values
        self.keys = np.array(keys)

    @overload
    def __getitem__(self, key: IndexerKey) -> IndexerValue: ...

    @overload
    def __getitem__(self, key: slice) -> ArrayLike[IndexerValue]: ...

    def __getitem__(
        self, key: IndexerKey | slice
    ) -> IndexerValue | ArrayLike[IndexerValue]:
        if isinstance(key, slice):
            start_index, stop_index = np.searchsorted(self.keys, [key.start, key.stop])
            result = self.values[start_index:stop_index]
            return result

        index = np.searchsorted(self.keys, key)
        if self.keys[index] != key:
            raise IndexError()
        return self.values[int(index)]


def index_key_to_absolute_indices(key: int | slice, obj: Sized) -> tuple[int, int, int]:
    """Convert an integer key or slice into it's absolute indices

    This will avoid calling len() on the object unless necessary

    Args:
        key(int | slice): an integer or slice
        obj(Sized): the object that the slice applies to (needed for negative indices)

    Examples:
        >>> index_key_to_absolute_indices(-1, 'abc')
        2
        >>> index_key_to_absolute_indices(slice(-5, None), 'abcdefgh')
        (3, 8, 1)

    """
    step = 1
    if isinstance(key, slice):
        start, stop, step = (
            key.start,
            key.stop,
            1 if key.step is None else key.step,
        )
    else:
        try:
            key = int(key)
        except TypeError as e:
            raise TypeError(f"key must be int or slice, not {type(key)}") from e
        start, stop = key, key + 1
        if key < 0:
            start = len(obj) + key
            stop = start + 1

    if start is None:
        start = 0
    if start < 0:
        start = len(obj) + start
    if stop is None:
        stop = len(obj)
    if stop < 0:
        stop = len(obj) + stop

    return start, stop, step
