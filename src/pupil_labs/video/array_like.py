from collections.abc import Iterator
from typing import (
    Protocol,
    TypeVar,
    overload,
)

ArrayLikeItemType = TypeVar("ArrayLikeItemType", covariant=True)


class ArrayLike(Protocol[ArrayLikeItemType]):
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int, /) -> ArrayLikeItemType: ...
    @overload
    def __getitem__(self, key: slice, /) -> "ArrayLike[ArrayLikeItemType]": ...
    def __getitem__(
        self, key: int | slice, /
    ) -> ArrayLikeItemType | "ArrayLike[ArrayLikeItemType]": ...
    def __iter__(self) -> Iterator[ArrayLikeItemType]: ...
