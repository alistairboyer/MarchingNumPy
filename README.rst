MarchingCubesLorensen.py
    An implementation of the marching cubes algorithm using numpy
    This follows the orignal Loresnen method and there is no ambiguity resolution
MarchingSquares.py
    An implementation of the marching squares algorithm using numpy
    Ambiguity resolution is performed
MarchingTriangles.py
    A reimagniation of the 2D marching algorithm as triangles instead of squares using numpy
    Ambiguity resolution is not required
DualContouring.py
    An implementation of dual contouring algorithm using numpy

Marching.py
    The marching process is consistent for each of the methods:
        - MarchingCubesLorensen
        - MarchingSquares
        - MarchingTriangles
    The process involves:
        - Finding intersects along volume axes
            this is handled by FindIntersects.py
        - Classifying the volume according the value at neighboring points
            this is handled by VolumeTypes.py
        - Reclassifying the volume for ambiguous cases by performing further tests
            this is handled by VolumeTypes.py
        - Looking up geometry from tables based upon the volume types
            this is handled by LookUpGeometry.py

FindIntersects.py
VolumeTypes.py
ResolveAmbiguous.py
LookUpGeometry.py

Checking.py
    A utility for performing validation on objects used by this module.
    Checks numpy arrays size and shape.


LaplacianSmoothing.py
    #