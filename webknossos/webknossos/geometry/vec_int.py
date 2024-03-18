import re
from operator import add, floordiv, mod, mul, sub
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np

VALUE_ERROR = "VecInt can be instantiated with int values `VecInt(1,2,3,4) or with `VecIntLike` object `VecInt([1,2,3,4])."

T = TypeVar("T", bound="VecInt")


class VecInt(tuple):
    """
    Class to represent an ND vector of integer values.
    """

    axes: Tuple[str, ...]
    _x_pos: Optional[int]
    _y_pos: Optional[int]
    _z_pos: Optional[int]

    def __new__(
        cls,
        *args: Union["VecIntLike", Iterable[str], int],
        axes: Optional[Iterable[str]] = None,
        **kwargs: int,
    ) -> "VecInt":
        """
        Class to represent a ND vector. Inherits from tuple and provides useful
        methods and operations on top. The vector stores the axes names together with the values.

        A small usage example:

        ```python
        from webknossos import VecInt

        # To create a vector with 4 elements and axes x, y, z, t:
        vector_1 = VecInt(1, 2, 3, 4, axes=("x", "y", "z", "t"))
        # or
        vector_1 = VecInt([1, 2, 3, 4], axes=("x", "y", "z", "t"))
        # or
        vector_1 = VecInt(x=1, y=2, z=3, t=4)

        vector_2 = VecInt.full(1, axes=("x", "y", "z", "t"))
        assert vector_2[0] == vector_2[1] == vector_2[2] == vector_2[3]

        assert vector_1 + vector_2 == VecInt(2, 3, 4, 5)
        ```
        """
        as_tuple: Optional[Tuple[int, ...]] = None

        if args:
            if isinstance(args[0], VecInt):
                return args[0]
            if isinstance(args[0], np.ndarray):
                assert np.count_nonzero(args[0] % 1) == 0, VALUE_ERROR
            if isinstance(args[0], str):
                return cls.from_str(args[0])
            if isinstance(args[0], Iterable):
                as_tuple = tuple(int(item) for item in args[0])
                if args[1:] and isinstance(args[1], Iterable):
                    assert all(isinstance(arg, str) for arg in args[1]), VALUE_ERROR
                    axes = tuple(args[1])  # type: ignore
            elif isinstance(args, Iterable):
                as_tuple = tuple(int(arg) for arg in args)  # type: ignore
            else:
                raise ValueError(VALUE_ERROR)
            assert axes is not None, VALUE_ERROR
        else:
            assert kwargs, VALUE_ERROR
            assert axes is None, VALUE_ERROR
            as_tuple = tuple(kwargs.values())

        assert as_tuple is not None, VALUE_ERROR

        self = super().__new__(cls, cast(Iterable, as_tuple))
        # self.axes is set in __new__ instead of __init__ so that pickling/unpickling
        # works without problems. As long as the deserialization of a tree instance
        # is not finished, the object is only half-initialized. Since self.axes
        # is needed after deepcopy, an error would be raised otherwise.
        # Also see:
        # https://stackoverflow.com/questions/46283738/attributeerror-when-using-python-deepcopy
        self.axes = tuple(axes or kwargs.keys())
        self._x_pos = self.axes.index("x") if "x" in self.axes else None
        self._y_pos = self.axes.index("y") if "y" in self.axes else None
        self._z_pos = self.axes.index("z") if "z" in self.axes else None

        return self

    def __getnewargs__(self) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        return (self.to_tuple(), self.axes)

    @property
    def x(self) -> int:
        """
        Returns the x component of the vector.
        """
        if self._x_pos is not None:
            return self[self._x_pos]

        raise ValueError("The vector does not have an x component.")

    @property
    def y(self) -> int:
        """
        Returns the y component of the vector.
        """
        if self._y_pos is not None:
            return self[self._y_pos]

        raise ValueError("The vector does not have an y component.")

    @property
    def z(self) -> int:
        """
        Returns the z component of the vector.
        """
        if self._z_pos is not None:
            return self[self._z_pos]

        raise ValueError("The vector does not have an z component.")

    @staticmethod
    def from_str(string: str) -> "VecInt":
        """
        Returns a new ND Vector from a string representation.

        Args:
            string (str): The string representation of the vector.

        Returns:
            VecInt: The new vector.
        """
        return VecInt(tuple(map(int, re.findall(r"\d+", string))))

    def with_replaced(self: T, index: int, new_element: int) -> T:
        """Returns a new ND Vector with a replaced element at a given index."""

        return self.__class__(
            *self[:index], new_element, *self[index + 1 :], axes=self.axes
        )

    def to_np(self) -> np.ndarray:
        """
        Returns the vector as a numpy array.
        """
        return np.array(self)

    def to_list(self) -> List[int]:
        """
        Returns the vector as a list.
        """
        return list(self)

    def to_tuple(self) -> Tuple[int, ...]:
        """
        Returns the vector as a tuple.
        """
        return tuple(self)

    def contains(self, needle: int) -> bool:
        """
        Checks if the vector contains a given element.
        """
        return any(element == needle for element in self)

    def is_positive(self, strictly_positive: bool = False) -> bool:
        """
        Checks if all elements in the vector are positive.

        Args:
            strictly_positive (bool): If True, checks if all elements are strictly positive.

        Returns:
            bool: True if all elements are positive, False otherwise.
        """
        if strictly_positive:
            return all(i > 0 for i in self)

        return all(i >= 0 for i in self)

    def is_uniform(self) -> bool:
        """
        Checks if all elements in the vector are the same.
        """
        first = self[0]
        return all(element == first for element in self)

    def _element_wise(
        self: T, other: Union[int, "VecIntLike"], fn: Callable[[int, Any], int]
    ) -> T:
        if isinstance(other, int):
            other_imported = VecInt.full(other, axes=self.axes)
        else:
            other_imported = VecInt(other, axes=self.axes)
            assert len(other_imported) == len(
                self
            ), f"{other} and {self} are not equally shaped."
        return self.__class__(
            **{
                axis: fn(self[i], other_imported[i]) for i, axis in enumerate(self.axes)
            },
            axes=None,
        )

    # Note: When adding regular tuples the first tuple is extended with the second tuple.
    # For VecInt we want to add the elements at the same index.
    # Do not add VecInt to plain tuple! Hence the type:ignore)
    def __add__(self: T, other: Union[int, "VecIntLike"]) -> T:  # type: ignore[override]
        return self._element_wise(other, add)

    def __sub__(self: T, other: Union[int, "VecIntLike"]) -> T:
        return self._element_wise(other, sub)

    # Note: When multiplying regular tuples with an int those are repeated,
    # which is a different behavior in the superclass! Hence the type:ignore.
    def __mul__(self: T, other: Union[int, "VecIntLike"]) -> T:  # type: ignore[override]
        return self._element_wise(other, mul)

    def __floordiv__(self: T, other: Union[int, "VecIntLike"]) -> T:
        return self._element_wise(other, floordiv)

    def __mod__(self: T, other: Union[int, "VecIntLike"]) -> T:
        return self._element_wise(other, mod)

    def __neg__(self: T) -> T:
        return self.__class__((-elem for elem in self), axes=self.axes)

    def ceildiv(self: T, other: Union[int, "VecIntLike"]) -> T:
        """
        Returns a new VecInt with the ceil division of each element by the other.
        """
        return (self + other - 1) // other

    def pairmax(self: T, other: Union[int, "VecIntLike"]) -> T:
        """
        Returns a new VecInt with the maximum of each pair of elements from the two vectors.
        """
        return self._element_wise(other, max)

    def pairmin(self: T, other: Union[int, "VecIntLike"]) -> T:
        """
        Returns a new VecInt with the minimum of each pair of elements from the two vectors.
        """
        return self._element_wise(other, min)

    def prod(self) -> int:
        """
        Returns the product of all elements in the vector.
        """
        return int(np.prod(self.to_np()))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({','.join((str(element) for element in self))})"
        )

    def add_or_none(self: T, other: Optional["VecInt"]) -> Optional[T]:
        """
        Adds two VecInts or returns None if the other is None.

        Args:
            other (Optional[VecInt]): The other vector to add.

        Returns:
            Optional[VecInt]: The sum of the two vectors or None if the other is None.
        """
        return None if other is None else self + other

    def moveaxis(
        self: T, source: Union[int, List[int]], target: Union[int, List[int]]
    ) -> T:
        """
        Allows to move one element at index `source` to another index `target`. Similar to
        np.moveaxis, this is *not* a swap operation but instead it moves the specified
        source so that the other elements move when necessary.

        Args:
            source (Union[int, List[int]]): The index of the element to move.
            target (Union[int, List[int]]): The index where the element should be moved to.

        Returns:
            VecInt: A new vector with the moved element.
        """

        # Piggy-back on np.moveaxis by creating an auxiliary array where the indices 0, 1 and
        # 2 appear in the shape.
        indices = np.moveaxis(
            np.zeros(tuple(i for i in range(len(self)))), source, target
        ).shape
        arr = self.to_np()[np.array(indices)]
        axes = np.array(self.axes)[np.array(indices)]
        return self.__class__(arr, axes=axes)

    @classmethod
    def zeros(cls, axes: Tuple[str, ...]) -> "VecInt":
        """
        Returns a new ND Vector with all elements set to 0.

        Args:
            length (int): The length of the vector.

        Returns:
            VecInt: The new vector.
        """
        return cls((0 for _ in range(len(axes))), axes=axes)

    @classmethod
    def ones(cls, axes: Tuple[str, ...]) -> "VecInt":
        """
        Returns a new ND Vector with all elements set to 1.

        Args:
            length (int): The length of the vector.

        Returns:
            VecInt: The new vector.
        """
        return cls((1 for _ in range(len(axes))), axes=axes)

    @classmethod
    def full(cls, an_int: int, axes: Tuple[str, ...]) -> "VecInt":
        """
        Returns a new ND Vector with all elements set to the same value.

        Args:
            an_int (int): The value of the elements.
            length (int): The length of the vector.

        Returns:
            VecInt: The new vector.
        """
        return cls((an_int for _ in range(len(axes))), axes=axes)


VecIntLike = Union[VecInt, Tuple[int, ...], np.ndarray, Iterable[int]]
