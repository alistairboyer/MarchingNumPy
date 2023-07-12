from __future__ import annotations
from typing import Dict, Tuple, Collection, Any, Optional
from numpy.typing import NDArray

import numpy
import cupy


def resolve_ambiguous_types(
    types: NDArray[numpy.integer[Any]],
    interpolated_face_values: NDArray[numpy.bool_],
    ambiguity_resolution: Dict[int, Collection[Any]],
) -> None:
    """
    Check for ambiguous cases and update types based upon
    :attr:`interpolated_face_values`.

    :attr:`ambiguity_resolution` is a ``dict`` that has the
    type from a corner value lookup as keys.
    The values are a list: [`test_values`, `test_selection`, `resolved_type`]
    where the `test_values` wil be checked against the
    :attr:`interpolated_face_values` and if they match,
    filtered by where the `test_selection` pattern is True,
    then the type is updated to be `resolved_type`.

    Args
    ----
        types: NDArray[numpy.integer[Any]]
            The types from corner value lookup.
            This is modified in place.
        interpolated_face_values: NDArray[numpy.bool_]
            Results of a test based upon volume values.
        ambiguity_resolution: Dict[int, Collection[Any]]
            Instructions on how to test for and
            resolve any ambiguity.

    """
    ambiguous_type: int
    resolutions: Collection[Any]
    for ambiguous_type, resolutions in ambiguity_resolution.items():
        filter: Tuple[Optional[NDArray[numpy.int64]], ...]
        filter = cupy.nonzero(types == ambiguous_type)

        if len(filter[0]) == 0:
            continue

        new_values = cupy.full_like(types[filter], ambiguous_type)

        for [test_values, test_selection, resolved_type] in resolutions:
            types[filter][interpolated_face_values[filter].flatten()] = resolved_type
