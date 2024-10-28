from typing import Iterator, Sequence, Sized, overload

from .array_like import ArrayLike, T
from .reader import index_key_to_indices


class LengthLookup(dict):
    def __missing__(self, key: Sized) -> int:
        return len(key)


class MultiArrayLike(ArrayLike[T]):
    def __init__(self, parts: Sequence[ArrayLike[T]]) -> None:
        self.parts = parts
        self.part_lengths = LengthLookup()

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> ArrayLike[T]: ...

    def __getitem__(self, key: int | slice) -> T | ArrayLike[T]:
        if isinstance(key, int):
            index = index_key_to_indices(key, self)[0]
            offset = 0
            for part in self.parts:
                part_length = self.part_lengths[part]
                part_index = index - offset
                if part_index < part_length:
                    return part[part_index]
                offset += part_length

            return self.parts[part_index][part_index]
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts)

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self)):
            yield self[i]
