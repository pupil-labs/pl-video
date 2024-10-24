from typing import Sequence, SupportsIndex, overload

import numpy as np

from .reader import index_key_to_indices
from .sequence import ArrayLike, T

# T = TypeVar("T")


class MultiArrayLike(ArrayLike[T]):
    def __init__(self, arrays: Sequence[ArrayLike[T]]) -> None:
        self.arrays = arrays
        self._start_indices = np.cumsum([0] + [len(part) for part in self.arrays])

    @overload
    def __getitem__(self, key: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[T]: ...

    def __getitem__(self, key: SupportsIndex | slice) -> T | ArrayLike[T]:
        if isinstance(key, int):
            index = index_key_to_indices(key, self)[0]
            if index >= len(self):
                raise IndexError("Index out of range.")

            part_index = (
                np.searchsorted(self._start_indices, index, side="right").item() - 1
            )
            part_key = int(index - self._start_indices[part_index])
            return self.arrays[part_index][part_key]
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return sum(len(part) for part in self.arrays)
