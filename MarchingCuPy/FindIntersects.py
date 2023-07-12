from __future__ import annotations
from typing import Tuple, Collection, Set, Any, Optional
from numpy.typing import NDArray, ArrayLike

import numpy
import cupy

from .Types import Intersect, Intersect_id

INTERPOLATION_VALUES: Set[str] = {"LINEAR", "HALFWAY", "COSINE"}


def find_intersects(
    slice_indexes: Any,
    volume: NDArray[Any],
    size_multiplier: NDArray[Any],
    volume_test: Optional[NDArray[numpy.bool_]] = None,
    level: ArrayLike = 0.0,
    interpolation: str = "LINEAR",
) -> Tuple[NDArray[Intersect], NDArray[Intersect_id]]:
    r"""Calculate intersects in volume data.

    This function considers values in a **volume** along slices
    as determined by **slice_indexes**.
    Each intersect where the **volume** crosses the **level**
    is recorded in the **intersects** NDArray.
    Each intersect has an associated unique id that
    is recorded in the **intersect_ids** NDArray.

    The **level** is used to generate a
    zero-referenced volume and volume_test as below.
    If **level** is non-truthy (i.e. `0.0` or `None`) then
    **volume** is assumed to be zero-referenced.
    If **volume_test** is already calculated then this can be passed in
    and, as long as **level** is non-truthy,
    this part of the calculation will be skipped. ::

        volume = volume - level
        volume_test = volume >= 0

    The crossing point is calculated according to value of **interpolation**:

    "HALFWAY"
        The crossing point is marked at 0.5 between the two corners.
        This is a fast interpolation but gives geometry with a block appearence.

    "LINEAR"
        The crossing point is interpolated using linear interpolation.

        .. math::
            \begin{align*}
            y &= mx + c \\
            y &= (v_{n+1}-v_{n})x + v_{n} \\
            x &= \frac{y - v_{n}}{v_{n+1} - v_{n}}
            \end{align*}

        At the zeroed volume crossing point, :math:`y = 0`

        .. math::
            x = \frac{v_{n}}{v_{n} - v_{n+1}}

    "COSINE"
        The crossing point is interpolated using cosine interpolation.

        .. math::
            \begin{align*}
            y &= \frac{h}{2}\cos{(\pi x)} + c \\
            y &= \frac{v_{n}-v_{n+1}}{2} \cos({\pi x}) + \frac{v_{n} + v_{n+1}}{2} \\
            x &= \frac{1}{\pi}\arccos\left({\frac{2y - v_{n} - v_{n+1}}{v_{n}-v_{n+1}}}\right)
            \end{align*}

        At the zeroed volume crossing point, :math:`y = 0`

        .. math::
            x = \frac{1}{\pi}\arccos\left({\frac{v_{n+1} + v_{n}}{v_{n+1}-v_{n}}}\right)

    Parameters
    ----------
    slice_indexes : Collection[Tuple[slice, ...], ...]
        A Collection of Tuple of Slices that describe where
        to look for the intersects.

    volume : ArrayLike
        A n-dimensional array containing the volume data for testing.
        Passed through :func:`cupy.asarray`.

    size_multiplier : NDArray[Any]
        A multiplier to convert a coordinate and direction into a unique id.

    volume_test : NDArray[numpy.bool\_], default = None
        Optional bool-like results of the results of testing the volume against the :attr:`level`.
        The size must match the size of :attr:`volume`.

    level : ArrayLike, default = 0.0
        The level against which volume is evaluated.
        Passed through :func:`cupy.asarray` if truthy.

    interpolation : {"LINEAR", "HALFWAY", "COSINE"}
        Interpolation of the crossing point.

    Returns
    -------
        intersects : NDArray[Intersect]
            A NDArray containing intersects, ordered by axis.
        intersect_ids : NDArray[Intersect_id]
            A NDArray containing a unique id for each intersect in same order as intersects for indexing.


    Raises
    ------
    ValueError
        If the :attr:`volume` is zero-sized; or
        if the shape of the supplied :attr:`volume_test` is not consistent with :attr:`volume`; or
        if :attr:`interpolation` is an invalid value.

    """

    volume = cupy.asarray(volume)
    if not volume.size > 0:
        raise ValueError("There must be at least one value is each supplied dimension.")
    if level:
        volume = volume - cupy.asarray(level)
    if volume_test is None or level:
        volume_test = (volume >= 0.0).astype(numpy.bool_)
    else:
        volume_test = cupy.asarray(volume_test, dtype=numpy.bool_)
        if not volume_test.shape == volume.shape:
            raise ValueError("The volume_test shape must match the volume shape.")
    interpolation = str(interpolation).upper()
    if interpolation not in INTERPOLATION_VALUES:
        raise ValueError(
            f'interpolation must be one of: {"; ".join(INTERPOLATION_VALUES)}'
        )

    nD: int = volume.ndim  # number of dimensions

    intersects: NDArray[Intersect]
    intersects = cupy.zeros((0, nD), dtype=Intersect)
    intersect_ids: NDArray[Intersect_id]
    intersect_ids = cupy.zeros((0,), dtype=Intersect_id)

    i: int
    n_slices: Tuple[slice, ...]
    nplus1_slices: Tuple[slice, ...]
    for i, (n_slices, nplus1_slices) in enumerate(slice_indexes):

        value_filter: Tuple[NDArray[numpy.int64], ...]  # Type from numpy.nonzero
        value_filter = cupy.nonzero(volume_test[n_slices] != volume_test[nplus1_slices])

        intersect_indices: NDArray[Intersect_id]
        intersect_indices = cupy.asarray(value_filter, dtype=Intersect_id).transpose()

        intersects_along_axis: NDArray[Intersect]
        intersects_along_axis = intersect_indices.astype(Intersect)

        if intersects_along_axis.size == 0:
            continue

        interpolated_offset: NDArray[Intersect]
        if interpolation == "HALFWAY":
            interpolated_offset = cupy.asarray(0.5, dtype=Intersect)

        else:
            n_values: NDArray[Any]
            nplus1_values: NDArray[Any]
            n_values = volume[n_slices][value_filter]
            nplus1_values = volume[nplus1_slices][value_filter]

            if interpolation == "LINEAR":
                interpolated_offset = (n_values / (n_values - nplus1_values)).astype(
                    Intersect
                )

            if interpolation == "COSINE":
                interpolated_offset = (
                    cupy.arccos(
                        (nplus1_values + n_values) / (nplus1_values - n_values)
                    )
                    / numpy.pi
                ).astype(Intersect)

        slice_vector: NDArray[numpy.integer[Any]]
        slice_vector = vector_from_slices(n_slices, nplus1_slices, absolute=True)

        interpolated_offset = (
            interpolated_offset
            * cupy.full_like(
                intersect_indices,
                slice_vector,
                dtype=Intersect,
            ).transpose()
        ).transpose()

        intersects_along_axis += interpolated_offset

        intersects = cupy.concatenate((intersects, intersects_along_axis), axis=0)

        intersect_ids = cupy.concatenate(
            (intersect_ids, (intersect_indices * size_multiplier).sum(axis=1) + i),
            axis=0,
        )


    return intersects, intersect_ids


def vector_from_slices(
    from_slices: Collection[slice],
    to_slices: Collection[slice],
    absolute: bool = False,
):
    """
    Calculates a direction vector based
    upon two Collections of slices.

    Args:
        from_slices (Collection[slice]): from Collection of slices.
        to_slices (Collection[slice]): to Collection of slices.
        absolute (bool): If True returns absolute values. Default False.

    Returns:
        NDArray[numpy.int8]: vector representing direction.
    """
    if absolute:
        return cupy.asarray(
            [
                abs(int(to_slice.start or 0) - int(from_slice.start or 0))
                for from_slice, to_slice in zip(from_slices, to_slices)
            ],
            dtype=numpy.int8,
        )
    return cupy.asarray(
        [
            int(to_slice.start or 0) - int(from_slice.start or 0)
            for from_slice, to_slice in zip(from_slices, to_slices)
        ],
        dtype=numpy.int8,
    )
