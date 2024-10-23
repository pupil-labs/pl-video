from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import numpy as np

from .reader import index_key_to_indices

T = TypeVar("T")


@dataclass
class MultiSequence(Generic[T], Sequence[T]):
    sequences: Sequence[Sequence[T]]

    def __post_init__(self) -> None:
        self._start_indices = np.cumsum([0] + [len(part) for part in self.sequences])

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[T]: ...

    def __getitem__(self, key: int | slice) -> T | Sequence[T]:
        if isinstance(key, int):
            index = index_key_to_indices(key, self)[0]
            if index >= len(self):
                raise IndexError("Index out of range.")

            part_index = (
                np.searchsorted(self._start_indices, index, side="right").item() - 1
            )
            part_key = int(index - self._start_indices[part_index])
            return self.sequences[part_index][part_key]
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return sum(len(part) for part in self.sequences)
