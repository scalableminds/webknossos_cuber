import re
from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np

from .vec_int import VecInt

VALUE_ERROR = "Vector components must be three integers or a Vec3IntLike object."


class Vec3Int(VecInt):
    def __new__(
        cls,
        vec: Union[int, "Vec3IntLike"],
        *args: int,
        y: Optional[int] = None,
        z: Optional[int] = None,
        **kwargs: int,
    ) -> "Vec3Int":
        """
        Class to represent a 3D vector. Inherits from tuple and provides useful
        methods and operations on top.

        A small usage example:

        ```python
        from webknossos import Vec3Int

        vector_1 = Vec3Int(1, 2, 3)
        vector_2 = Vec3Int.full(1)
        assert vector_2.x == vector_2.y == vector_2.y

        assert vector_1 + vector_2 == (2, 3, 4)
        ```
        """

        if isinstance(vec, Vec3Int):
            return vec

        as_tuple = super().__new__(cls, vec, *args, y=y, z=z, **kwargs)
        assert as_tuple is not None and len(as_tuple) == 3, VALUE_ERROR

        return cast(Vec3Int, as_tuple)

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    @property
    def z(self) -> int:
        return self[2]

    def with_x(self, new_x: int) -> "Vec3Int":
        return Vec3Int.from_xyz(new_x, self.y, self.z)

    def with_y(self, new_y: int) -> "Vec3Int":
        return Vec3Int.from_xyz(self.x, new_y, self.z)

    def with_z(self, new_z: int) -> "Vec3Int":
        return Vec3Int.from_xyz(self.x, self.y, new_z)

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    @staticmethod
    def from_xyz(x: int, y: int, z: int) -> "Vec3Int":
        """Use Vec3Int.from_xyz for fast construction."""

        # By calling __new__ of tuple directly, we circumvent
        # the tolerant (and potentially) slow Vec3Int.__new__ method.
        return tuple.__new__(Vec3Int, (x, y, z))

    @staticmethod
    def from_vec3_float(vec: Tuple[float, float, float]) -> "Vec3Int":
        return Vec3Int(int(vec[0]), int(vec[1]), int(vec[2]))

    @staticmethod
    def from_vec_or_int(vec_or_int: Union["Vec3IntLike", int]) -> "Vec3Int":
        if isinstance(vec_or_int, int):
            return Vec3Int.full(vec_or_int)

        return Vec3Int(vec_or_int)

    @staticmethod
    def from_str(string: str) -> "Vec3Int":
        if re.match(r"\(\d+,\d+,\d+\)", string):
            return Vec3Int(tuple(map(int, re.findall(r"\d+", string))))

        return Vec3Int.full(int(string))

    @classmethod
    def zeros(cls, length: int = 3) -> "Vec3Int":
        del length

        return cls(0, 0, 0)

    @classmethod
    def ones(cls, length: int = 3) -> "Vec3Int":
        del length

        return cls(1, 1, 1)

    @classmethod
    def full(cls, an_int: int, length: int = 3) -> "Vec3Int":
        del length

        return cls(an_int, an_int, an_int)


Vec3IntLike = Union[Vec3Int, Tuple[int, int, int], np.ndarray, Iterable[int]]
