from collections.abc import Sequence
from typing import Generic, TypeVar, overload

FrameType = TypeVar("FrameType")


class FrameSlice(Generic[FrameType], Sequence[FrameType]):
    def __init__(self, target: Sequence[FrameType], slice_value: slice):
        self.target = target
        self.slice = slice_value
        self.start, self.stop, self.step = slice_value.indices(len(self.target))

    @overload
    def __getitem__(self, key: int) -> FrameType: ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[FrameType]: ...

    def __getitem__(self, key: int | slice) -> FrameType | Sequence[FrameType]:
        if isinstance(key, int):
            if key > len(self) - 1:
                raise IndexError()
            return self.target[key + self.start]
        elif isinstance(key, slice):
            # TODO(dan): implement FrameSlice(self, new_slice)
            raise NotImplementedError()
        else:
            raise TypeError

    def __len__(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.target})"
            "["
            f"{'' if self.slice.start is None else self.slice.start}"
            ":"
            f"{'' if self.slice.stop is None else self.slice.stop}"
            "]"
        )
