from collections.abc import Iterator
from typing import Generic, TypeVar, overload

from pupil_labs.video.array_like import ArrayLike
from pupil_labs.video.constants import LAZY_FRAME_SLICE_LIMIT
from pupil_labs.video.indexing import index_key_to_absolute_indices

FrameType = TypeVar("FrameType")


class FrameSlice(Generic[FrameType]):
    def __init__(
        self,
        target: ArrayLike[FrameType],
        slice_value: slice,
        lazy_frame_slice_limit: int = LAZY_FRAME_SLICE_LIMIT,
    ):
        self.target = target
        self.slice = slice_value
        self.start, self.stop, self.step = index_key_to_absolute_indices(
            slice_value, self.target
        )
        self.lazy_frame_slice_limit = lazy_frame_slice_limit

    @overload
    def __getitem__(self, key: int) -> FrameType: ...

    @overload
    def __getitem__(self, key: slice) -> "FrameSlice[FrameType] | list[FrameType]": ...

    def __getitem__(
        self, key: int | slice
    ) -> "FrameType | FrameSlice[FrameType] | list[FrameType]":
        if isinstance(key, slice):
            start, stop, step = index_key_to_absolute_indices(key, self)
            new_slice = slice(
                self.start + start,
                self.start + min(stop, len(self)),
                step,
            )
            frameslice = FrameSlice(
                self.target,
                new_slice,
                lazy_frame_slice_limit=self.lazy_frame_slice_limit,
            )
            if len(frameslice) < self.lazy_frame_slice_limit:
                return list(frameslice)
            return frameslice
        else:
            try:
                key = int(key)
            except TypeError as e:
                raise TypeError(f"key must be slice or int not: {type(key)}") from e
            if key > len(self) - 1:
                raise IndexError()
            return self.target[key + self.start]

    def __len__(self) -> int:
        length = self.stop - self.start
        return 0 if length < 0 else length

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

    def __iter__(self) -> Iterator[FrameType]:
        for i in range(len(self)):
            yield self[i]
