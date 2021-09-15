import re
from functools import total_ordering
from math import log2
from typing import Any, List

import numpy as np


@total_ordering
class Mag(object):
    def __init__(self, mag: Any):
        self._mag: List[int] = []

        if isinstance(mag, int):
            self._mag = [mag] * 3
        elif isinstance(mag, list):
            self._mag = mag
        elif isinstance(mag, tuple):
            self._mag = [mag_d for mag_d in mag]
        elif isinstance(mag, str):
            if re.match(r"^\d+$", mag) is not None:
                self._mag = [int(mag)] * 3
            elif re.match(r"^\d+-\d+-\d+$", mag) is not None:
                self._mag = [int(m) for m in mag.split("-")]
        elif isinstance(mag, Mag):
            self._mag = mag._mag
        elif isinstance(mag, np.ndarray):
            assert mag.shape == (3,)
            self._mag = list(mag)

        if self._mag is None or len(self._mag) != 3:
            raise ValueError(
                "magnification must be int or a vector3 of ints or a string shaped like e.g. 2-2-1"
            )

        for m in self._mag:
            assert log2(m) % 1 == 0, "magnification needs to be power of 2."

    def __lt__(self, other: Any) -> bool:
        return max(self._mag) < (max(Mag(other).to_list()))

    def __le__(self, other: Any) -> bool:
        return max(self._mag) <= (max(Mag(other).to_list()))

    def __eq__(self, other: Any) -> bool:
        return all(m1 == m2 for m1, m2 in zip(self._mag, Mag(other)._mag))

    def __str__(self) -> str:
        return self.to_layer_name()

    def __expr__(self) -> str:
        return f"Mag({self.to_layer_name()})"

    def to_layer_name(self) -> str:
        x, y, z = self._mag
        if x == y and y == z:
            return str(x)
        else:
            return self.to_long_layer_name()

    def to_long_layer_name(self) -> str:
        x, y, z = self._mag
        return "{}-{}-{}".format(x, y, z)

    def to_list(self) -> List[int]:
        return self._mag.copy()

    def to_np(self) -> np.ndarray:
        return np.array(self._mag)

    def scaled_by(self, factor: int) -> "Mag":
        return Mag([mag * factor for mag in self._mag])

    def divided(self, coord: List[int]) -> List[int]:
        return [c // m for c, m in zip(coord, self._mag)]

    def divided_by(self, d: int) -> "Mag":
        return Mag([mag // d for mag in self._mag])

    def __hash__(self) -> int:
        return hash(tuple(self._mag))
