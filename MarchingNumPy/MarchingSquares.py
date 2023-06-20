from __future__ import annotations
from typing import Tuple, List, Dict, Callable, Any
from numpy.typing import NDArray

import numpy

from . import Marching
from . import IndexingTools


marching_squares: Callable  # See end of file
"""
Convert a 2d :attr:`volume` into isolines
by splitting the volume into squares and evaluating
where the values in the volume cross a :attr:`level` threshold
along the edges of the squares.
Can perform a test where the volume type is ambiguous to resolve the ambiguity.

Args
----

volume : NDArray
    volume
level : float | NDArray
    level

Keyword Args
------------
interpolation : {'LINEAR', 'HALFWAY', 'COSINE'}
    Interpolation method for :func:`.FindIntersects.find_intersects`.
step_size : int
    Defaults to 1.
ambiguity_resolution : bool
    Perform ambiguity resolution.
    Defaults to ``True``.

Returns
-------
vertices : NDArray
geometry : NDArray

"""


VOLUME_TYPE_SLICES: Tuple[Tuple[slice, ...], ...]
VOLUME_TYPE_SLICES = IndexingTools.str_to_index_tuple(
    "[:-1, :-1]",  # 1: x,   y
    "[1: , :-1]",  # 2: x+1, y
    "[1: , 1: ]",  # 4: x+1, y+1
    "[:-1, 1: ]",  # 8: x,   y+1
)

INTERSECT_SLICE_INDEXES = [
    IndexingTools.str_to_index_tuple("[:-1, :]", "[1:, :]"),  # x,y - x+1,y
    IndexingTools.str_to_index_tuple("[:, :-1]", "[:, 1:]"),  # x,y - x,y+1
]


EDGE_DELTA: NDArray[numpy.uint8]
EDGE_DELTA = numpy.asarray(
    [
        [0, 0],  # 0 Bottom
        [1, 0],  # 1 Right
        [0, 1],  # 2 Top
        [0, 0],  # 3 Left
    ],
    dtype=numpy.uint8,
)

EDGE_DIRECTION: NDArray[numpy.uint8]
EDGE_DIRECTION = numpy.asarray(
    [
        0,  # 0
        1,  # 1
        0,  # 2
        1,  # 3
    ],
    dtype=numpy.uint8,
)

# Edge definitions
_NONE: List[int] = [-1, -1]
_BOTTOM_LEFT: List[int] = [0, 3]
_BOTTOM_RIGHT: List[int] = [1, 0]
_VERTICAL: List[int] = [2, 0]
_HORIZONTAL: List[int] = [1, 3]
_TOP_LEFT: List[int] = [2, 3]
_TOP_RIGHT: List[int] = [2, 1]

# Geometry lookup table
"""
Geometry lookup table for marching squares.

:meta private:
"""
GEOMETRY_LOOKUP: NDArray[numpy.int8]
GEOMETRY_LOOKUP = numpy.asarray(
    [
        # default values
        _NONE + _NONE,  # 0
        _BOTTOM_LEFT + _NONE,  # 1
        _BOTTOM_RIGHT + _NONE,  # 2
        _HORIZONTAL + _NONE,  # 3
        _TOP_RIGHT + _NONE,  # 4
        _TOP_RIGHT + _BOTTOM_LEFT,  # 5
        _VERTICAL + _NONE,  # 6
        _TOP_LEFT + _NONE,  # 7
        _TOP_LEFT[::-1] + _NONE,  # 8
        _VERTICAL[::-1] + _NONE,  # 9
        _TOP_LEFT[::-1] + _BOTTOM_RIGHT,  # 10
        _TOP_RIGHT[::-1] + _NONE,  # 11
        _HORIZONTAL[::-1] + _NONE,  # 12
        _BOTTOM_RIGHT[::-1] + _NONE,  # 13
        _BOTTOM_LEFT[::-1] + _NONE,  # 14
        _NONE + _NONE,  # 15
        # flipped values for ambiguous cases
        _TOP_LEFT + _BOTTOM_RIGHT[::-1],  # 5 -> 16
        _TOP_RIGHT[::-1] + _BOTTOM_LEFT[::-1],  # 10 -> 17
    ],
    dtype=numpy.int8,
)

SQUARE_AMBIGUITY_RESOLUTION: Dict[int, List[Any]]
SQUARE_AMBIGUITY_RESOLUTION = {
    5: [
        [False, True, 16],
    ],
    10: [
        [True, True, 17],
    ],
}


def interpolate_face_values(
    volume: NDArray[Any],
) -> NDArray[numpy.bool_]:
    # TODO only calculate this where appropriate instead of over entire volume
    n: int
    values: NDArray[numpy.bool_]
    values = numpy.zeros((*[n - 1 for n in volume.shape], 1), dtype=numpy.bool_)
    values[..., 0] = (
        volume[:-1, :-1] * volume[1:, 1:] < volume[:-1, 1:] * volume[1:, :-1]
    )
    return values


marching_squares = Marching.marching_factory(
    nD=2,
    nEdges=2,
    intersect_slice_indexes=INTERSECT_SLICE_INDEXES,
    volume_type_slices=VOLUME_TYPE_SLICES,
    ambiguity_resolution=SQUARE_AMBIGUITY_RESOLUTION,
    ambiguity_interpolater=interpolate_face_values,
    geometry_array=GEOMETRY_LOOKUP,
    edge_direction=EDGE_DIRECTION,
    edge_delta=EDGE_DELTA,
)
