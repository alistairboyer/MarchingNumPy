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
    dimXY = dimX * dimY
    size_multiplier = numpy.asarray([1, dimX, dimX * dimY]) * 3

    def calculate_vertex_id(x, y, z, direction):
        return (x + y * dimX + z * dimXY) * 3 + direction


    # find where volume crosses level -> vertex
    def find_all_vertices():
        filt = numpy.nonzero(volume_test[:-1, :, :] != volume_test[1:, :, :])
        corners = numpy.transpose(filt)
        vertex_idsX = (corners * size_multiplier).sum(axis=1) + DirectionX
        verticesX = corners.astype("float")
        interpolated_offset = interpolate(volume[:-1, :, :][filt], volume[1:, :, :][filt], level)
        verticesX[..., DirectionX] += interpolated_offset

        filt = numpy.nonzero(volume_test[:, :-1, :] != volume_test[:, 1:, :])
        corners = numpy.transpose(filt)
        vertex_idsY = (corners * size_multiplier).sum(axis=1) + DirectionY
        verticesY = corners.astype("float")
        interpolated_offset = interpolate(volume[:, :-1, :][filt], volume[:, 1:, :][filt], level)
        verticesY[..., DirectionY] += interpolated_offset

        filt = numpy.nonzero(volume_test[:, :, :-1] != volume_test[:, :, 1:])
        corners = numpy.transpose(filt)
        vertex_idsZ = (corners * size_multiplier).sum(axis=1) + DirectionZ
        verticesZ = corners.astype("float")
        interpolated_offset = interpolate(volume[:, :, :-1][filt], volume[:, :, 1:][filt], level)
        verticesZ[..., DirectionZ] += interpolated_offset

        vertex_ids_np = vertex_idsX.tolist() + vertex_idsY.tolist() + vertex_idsZ.tolist()
        vertices_np = verticesX.tolist() + verticesY.tolist() + verticesZ.tolist()
        return vertex_ids_np, vertices_np

    vertex_ids_np, vertices_np = find_all_vertices()


    # calculate volume types
    def calculate_volume_types():
        volume_types = numpy.zeros(tuple(n - 1 for n in volume_test.shape), dtype="uint8")
        numpy.bitwise_or(volume_types, 1 << 0, out=volume_types, where=volume_test[:-1, :-1, :-1])
        numpy.bitwise_or(volume_types, 1 << 1, out=volume_types, where=volume_test[1:, :-1, :-1])
        numpy.bitwise_or(volume_types, 1 << 2, out=volume_types, where=volume_test[1:, 1:, :-1])
        numpy.bitwise_or(volume_types, 1 << 3, out=volume_types, where=volume_test[:-1, 1:, :-1])
        numpy.bitwise_or(volume_types, 1 << 4, out=volume_types, where=volume_test[:-1, :-1, 1:])
        numpy.bitwise_or(volume_types, 1 << 5, out=volume_types, where=volume_test[1:, :-1, 1:])
        numpy.bitwise_or(volume_types, 1 << 6, out=volume_types, where=volume_test[1:, 1:, 1:])
        numpy.bitwise_or(volume_types, 1 << 7, out=volume_types, where=volume_test[:-1, 1:, 1:])
        return volume_types
    volume_types = calculate_volume_types()


    # lookup geometry
    def lookup_all_geometry():
        triangle_ids_np = list()
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

            triangle_ids_np = triangle_ids_np + numpy.asarray((vertex_ids0, vertex_ids1, vertex_ids2)).transpose().tolist()
        return triangle_ids_np

    triangle_ids_np = lookup_all_geometry()


    # enumerate volume
    for z in range(dimZ):
        for y in range(dimY):
            for x in range(dimX):

                def edge_to_vertex_id(edge_number):
                    dx, dy, dz = EDGE_DELTA[edge_number]
                    direction = EDGE_DIRECTION[edge_number]
                    return calculate_vertex_id(x + dx, y + dy, z + dz, direction)

                # find where volume crosses level -> vertex
                def find_vertices():
                    if x < (dimX - 1) and volume_test[x, y, z] != volume_test[x + 1, y, z]:
                        delta = interpolate(volume[x, y, z], volume[x + 1, y, z], level)
                        vertices.append([x + delta, y, z])
                        vertex_ids.append(calculate_vertex_id(x, y, z, DirectionX))
                    if y < (dimY - 1) and volume_test[x, y, z] != volume_test[x, y + 1, z]:
                        delta = interpolate(volume[x, y, z], volume[x, y + 1, z], level)
                        vertices.append([x, y + delta, z])
                        vertex_ids.append(calculate_vertex_id(x, y, z, DirectionY))
                    if z < (dimZ - 1) and volume_test[x, y, z] != volume_test[x, y, z + 1]:
                        delta = interpolate(volume[x, y, z], volume[x, y, z + 1], level)
                        vertices.append([x, y, z + delta])
                        vertex_ids.append(calculate_vertex_id(x, y, z, DirectionZ))
                find_vertices()

                if x == (dimX - 1) or y == (dimY - 1) or z == (dimZ - 1):
                    continue

                # calculate volume type
                def calculate_volume_type():
                    volume_type = 0
                    if volume_test[x, y, z]:
                        volume_type |= 1 << 0
                    if volume_test[x + 1, y, z]:
                        volume_type |= 1 << 1
                    if volume_test[x + 1, y + 1, z]:
                        volume_type |= 1 << 2
                    if volume_test[x, y + 1, z]:
                        volume_type |= 1 << 3
                    if volume_test[x, y, z + 1]:
                        volume_type |= 1 << 4
                    if volume_test[x + 1, y, z + 1]:
                        volume_type |= 1 << 5
                    if volume_test[x + 1, y + 1, z + 1]:
                        volume_type |= 1 << 6
                    if volume_test[x, y + 1, z + 1]:
                        volume_type |= 1 << 7
                    return volume_type
                volume_type = calculate_volume_type()
                assert volume_type == volume_types[x, y, z]

                # lookup geometry
                def lookup_geometry():
                    lookup = GEOMETRY_LOOKUP[volume_type]
                    for i in range(0, len(lookup), 3):
                        if lookup[i] < 0:
                            break
                        edge0, edge1, edge2 = lookup[i : i + 3]
                        vertex_id0 = edge_to_vertex_id(edge0)
                        vertex_id1 = edge_to_vertex_id(edge1)
                        vertex_id2 = edge_to_vertex_id(edge2)
                        triangle_ids.append([vertex_id0, vertex_id1, vertex_id2])
                lookup_geometry()

    # convert ids to indexes
    def convert_indexes():
        order_of_ids = {id: order for order, id in enumerate(vertex_ids)}
        for triangle_corners in triangle_ids:
            triangles.append([order_of_ids[c] for c in triangle_corners])
    convert_indexes()

    return vertices, triangles


def profile(target, fn="/tmp/marching.profile"):
    import cProfile
    import pstats
    cProfile.run(target, fn)
    p = pstats.Stats(fn)
    p.strip_dirs().sort_stats(pstats.SortKey.NFL)
    p.print_stats("arching")

if __name__ == "__main__":
    import skimage

    volume = numpy.load("test_volume.npy")
    print(f"volume loaded with shape {volume.shape}")

    level = 0.05
    print(f"processing volume at level {level}")

    vertices, triangles = list(), list()
    profile("""vertices, triangles = marching(volume, level=level)""")
    print(f"marching: {len(vertices)} vertices, {len(triangles)} triangles found")

    skimage.measure.marching_cubes(volume, level=level, method="_lorensen")
    profile("""vertices_sk, triangles_sk = skimage.measure.marching_cubes(volume, level=level, method="_lorensen")""")
    vertices_sk = vertices_sk.tolist()
    print(f"skimage: {len(vertices_sk)} vertices, {len(triangles_sk)} triangles found")



