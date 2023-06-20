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

    def calculate_vertex_id(x, y, z, direction):
        return (x + y * dimX + z * dimXY) * 3 + direction

    # enumerate volume
    for z in range(dimZ):
        for y in range(dimY):
            for x in range(dimX):

                def edge_to_vertex_id(edge_number):
                    dx, dy, dz = EDGE_DELTA[edge_number]
                    direction = EDGE_DIRECTION[edge_number]
                    return calculate_vertex_id(x + dx, y + dy, z + dz, direction)

                # find where volume crosses level -> vertex
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

                if x == (dimX - 1) or y == (dimY - 1) or z == (dimZ - 1):
                    continue

                # calculate volume type
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

                # lookup geometry
                lookup = GEOMETRY_LOOKUP[volume_type]
                for i in range(0, len(lookup), 3):
                    if lookup[i] < 0:
                        break
                    edge0, edge1, edge2 = lookup[i : i + 3]
                    vertex_id0 = edge_to_vertex_id(edge0)
                    vertex_id1 = edge_to_vertex_id(edge1)
                    vertex_id2 = edge_to_vertex_id(edge2)
                    triangle_ids.append([vertex_id0, vertex_id1, vertex_id2])

    # convert ids to indexes
    order_of_ids = {id: order for order, id in enumerate(vertex_ids)}
    for triangle_corners in triangle_ids:
        triangles.append([order_of_ids[c] for c in triangle_corners])

    return vertices, triangles


if __name__ == "__main__":
    volume = numpy.load("test_volume.npy")
    print(f"volume loaded with shape {volume.shape}")

    for level in [0.05, -0.05, 0.005, -0.005]:
        print(f"processing volume at level {level}")
        vertices, triangles = marching(volume, level=level)

        import skimage

        vertices_sk, triangles_sk = skimage.measure.marching_cubes(volume, level=level, method="_lorensen")
        vertices_sk = vertices_sk.tolist()

        print(f"marching: {len(vertices)} vertices, {len(triangles)} triangles found")
        print(f"skimage: {len(vertices_sk)} vertices, {len(triangles_sk)} triangles found")
        print()

        numpy.savez(
                    f"/tmp/marching_{level}.npz",
                    vertices=vertices,
                    triangles=triangles,
                    vertices_sk=vertices_sk,
                    triangles_sk=triangles_sk,
                    )
