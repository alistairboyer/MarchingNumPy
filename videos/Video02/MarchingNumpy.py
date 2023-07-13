import numpy
from LorensenLookUpTable import (
    GEOMETRY_LOOKUP,
    EDGE_DELTA,
    EDGE_DIRECTION,
    DirectionX,
    DirectionY,
    DirectionZ,
)


def interpolate(a, b, level):
    # zero values to level
    a = a - level
    b = b - level
    # linear interpolation
    return a / (a - b)


def marching(volume, level=0.0):
    vertices = list()
    vertex_ids = list()
    triangles = list()
    triangle_ids = list()

    # compare volume to level
    volume_test = numpy.asarray(volume >= level, dtype="bool")
    dimX, dimY, dimZ = volume_test.shape
    size_multiplier = numpy.asarray([1, dimX, dimX * dimY]) * 3

    # find where volume crosses level -> vertex
    filt = numpy.nonzero(volume_test[:-1, :, :] != volume_test[1:, :, :])
    corners = numpy.transpose(filt)
    vertex_ids += ((corners * size_multiplier).sum(axis=1) + DirectionX).tolist()
    verticesX = corners.astype("float")
    interpolated_offset = interpolate(volume[:-1, :, :][filt], volume[1:, :, :][filt], level)
    verticesX[..., DirectionX] += interpolated_offset
    vertices += verticesX.tolist()

    filt = numpy.nonzero(volume_test[:, :-1, :] != volume_test[:, 1:, :])
    corners = numpy.transpose(filt)
    vertex_ids += ((corners * size_multiplier).sum(axis=1) + DirectionY).tolist()
    verticesY = corners.astype("float")
    interpolated_offset = interpolate(volume[:, :-1, :][filt], volume[:, 1:, :][filt], level)
    verticesY[..., DirectionY] += interpolated_offset
    vertices += verticesY.tolist()

    filt = numpy.nonzero(volume_test[:, :, :-1] != volume_test[:, :, 1:])
    corners = numpy.transpose(filt)
    vertex_ids += ((corners * size_multiplier).sum(axis=1) + DirectionZ).tolist()
    verticesZ = corners.astype("float")
    interpolated_offset = interpolate(volume[:, :, :-1][filt], volume[:, :, 1:][filt], level)
    verticesZ[..., DirectionZ] += interpolated_offset
    vertices += verticesZ.tolist()

    # calculate volume types
    volume_types = numpy.zeros(tuple(n - 1 for n in volume_test.shape), dtype="uint8")
    numpy.bitwise_or(volume_types, 1 << 0, out=volume_types, where=volume_test[:-1, :-1, :-1])
    numpy.bitwise_or(volume_types, 1 << 1, out=volume_types, where=volume_test[1:, :-1, :-1])
    numpy.bitwise_or(volume_types, 1 << 2, out=volume_types, where=volume_test[1:, 1:, :-1])
    numpy.bitwise_or(volume_types, 1 << 3, out=volume_types, where=volume_test[:-1, 1:, :-1])
    numpy.bitwise_or(volume_types, 1 << 4, out=volume_types, where=volume_test[:-1, :-1, 1:])
    numpy.bitwise_or(volume_types, 1 << 5, out=volume_types, where=volume_test[1:, :-1, 1:])
    numpy.bitwise_or(volume_types, 1 << 6, out=volume_types, where=volume_test[1:, 1:, 1:])
    numpy.bitwise_or(volume_types, 1 << 7, out=volume_types, where=volume_test[:-1, 1:, 1:])

    # lookup geometry
    lookup = GEOMETRY_LOOKUP[volume_types]
    for i in range(0, lookup.shape[-1], 3):
        filt = numpy.nonzero(lookup[..., i] >= 0)
        if filt[0].size == 0:
            break
        corners = numpy.transpose(filt)
        edge_info = lookup[filt][..., i]
        vertex_ids0 = ((corners + EDGE_DELTA[edge_info]) * size_multiplier).sum(axis=1) + EDGE_DIRECTION[edge_info]
        edge_info = lookup[filt][..., i + 1]
        vertex_ids1 = ((corners + EDGE_DELTA[edge_info]) * size_multiplier).sum(axis=1) + EDGE_DIRECTION[edge_info]
        edge_info = lookup[filt][..., i + 2]
        vertex_ids2 = ((corners + EDGE_DELTA[edge_info]) * size_multiplier).sum(axis=1) + EDGE_DIRECTION[edge_info]

        triangle_ids = triangle_ids + numpy.asarray((vertex_ids0, vertex_ids1, vertex_ids2)).transpose().tolist()

    # convert ids to indexes
    def convert_indexes():
        order_of_ids = {id: order for order, id in enumerate(vertex_ids)}
        for triangle_corners in triangle_ids:
            triangles.append([order_of_ids[c] for c in triangle_corners])
    convert_indexes()

    return vertices, triangles


if __name__ == "__main__":
    volume = numpy.load("test_volume.npy")
    print(f"volume loaded with shape {volume.shape}")

    level = 0.05
    print(f"processing volume at level {level}")

    vertices, triangles = marching(volume, level=level)
    print(f"marching: {len(vertices)} vertices, {len(triangles)} triangles found")
    #numpy.savez("/tmp/marching_numpy.pyz", vertices=vertices, triangles=triangles)

    #import skimage
    #vertices_sk, triangles_sk = skimage.measure.marching_cubes(volume, level=level, method="_lorensen")
