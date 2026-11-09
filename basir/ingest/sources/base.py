"""The FrameSource interface every frame provider implements."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from basir.core.types import CameraIntrinsics, Frame


@runtime_checkable
class FrameSource(Protocol):
    """Yields frames in time order. Datasets, phone recordings and live streams all fit."""

    @property
    def intrinsics(self) -> CameraIntrinsics: ...

    def __iter__(self) -> Iterator[Frame]: ...

    def __len__(self) -> int: ...

    def close(self) -> None: ...
