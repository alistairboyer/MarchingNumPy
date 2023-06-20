Background
==========


Data Visualisation
------------------
The visualisation of numeric data is a key component of computer graphics. 

Data is often encountered as numeric values distibuted over a regular space. 
Sometimes these volumes data are called voxels. 
Visualisation of this numeric data is key to the interpretation and/or appreciation of the data.

Selected examples:
 - Medicine: medical imaging such as MRI and CT scanners.
 - Science: quantum mechanical calculations and data recording.
 - Cartography: linking data such as height, land usage, etc to map coordinates. 
 - Entertainment: generation of artwork generated terrain and visualsation of metaballs.

An important method for the visualisation of volume data is to calculate an `isosurface`: 
a mesh surface that follows as closely as possible where the data 
crosses a certain `level` threshold. 

Marching Cubes: Lorensen
------------------------
This package includes an implementation of the Lorensen Marching Cubes algorithm
using only python and numpy methods: :meth:`.MarchingCubesLorensen.marching_cubes_lorensen`.

Lorensen and Cline described the `Marching Cubes` algorithm in 1987.

Marching Cubes joins all points where a `volume` of numeric values crosses 
a `level` threshold along each axis to create a triangular mesh. 
This algorithm considers each `unit` or `cube` of the data individually
and performs the following operations:

- **Calculate Intersects.**
   For each time that the volume value crosses the `level` threshold along each axis,
   interpolate the position of the intersection - this becomes a `vertex` for the output mesh. 

- **Assign Types.**
   Consider each corner of the volume `unit` and whether it is above or below the `level` threshold.
   Assign a unique identifier to each `unit` based upon the outcome of these logical binary tests.
   For a cube there are 8 vertices that gives 
   :math:`2^8 = 1\newcommand*\ShiftLeft{\ll}8 = 256`
   possible outcomes.
   However, these can be reduced to the same fundamental 14 types by symmetry.

- **Look Up Geometry.** 
   Use the calculated type to index a precalculated geometry table the defines 
   how the calculated vertices should be joined using triangular geometry.

Drawbacks:
   For intricate volume data, the simple reduction to 14 types of cube is insufficient to describe the isosurface.
   This ambiguity can lead to gaps within the mesh. 
Alternative Marching Approaches:
   - Marching Tetrahedra
   - Marching Cubes 33



Marching Cubes in 2D: Marching Squares and Marching Triangles
-------------------------------------------------------------
This package includes an implementation of the Marching Squares and Marching Triangles algorithms 
using numpy: :meth:`MarchingSquares.marching_squares` and
:meth:`MarchignTriangles.marching_triangles` / :meth:`MarchignTriangles.marching_triangles_reversed`.

The same logic can be applied 2-dimensionally to find `isolines` or `contours` for 2D values.
The calculation of intersects is the same but now in 2D along two axes rather than 3D. 
The marching squares have 4 vertices that gives :math:`2^4 = 1\newcommand*\ShiftLeft{\ll}4 = 16`
possible types that are reduced to 7 by symmetry.
It is trivial to draw up the look-up table by hand.
# Ambiguity.


Marching Triangles is an exercise where the squares are split into triangles along their diagonals.
Triangles can be split in eother of two directions / or \\.
This gives someThis results in a system with no ambiguity.



Dual Contouring
---------------
This package includes an implementation of the Dual Contouring algorithm using numpy: :func:`dual_contouring`.

`Dual Contouring` is an alternative approach to calculating an isosurface.
The approach is similar to `Marching Cubes` but instead of constructing a 
mesh joining the crossing points along the axes; it constructs a mesh 
between the centre of each volume `unit`. 
For each time that the volume value crosses the `level` threshold along each axis,
interpolate the position of the intersection - this becomes a `vertex` for the output mesh. 


Marching NumPy
--------------
The importance of calculating isosurfaces this was means there are multiple implementations
of the marching cubes algorithm.
Furthermore, the marching cubes algorithm is a common teaching tool in computer science. 

Within python, marching cubes is available from multiple sources. `[PiPy marching cubes] <https://pypi.org/search/?q="marching+cubes">`_

within the scikit-image package as :func:`skimage.measure.marching-cubes`
`API <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes>`_
`Description and example <https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html>`_

   -scikit-image
   -random person
   -random person2

NumPy is a powerful package for manipulating data in python that is used widely for scientific computing and data analysis.
NumPy provides a very efficient way to store arrayed data and a suite of functions to process the data. 
If NumPy functions are exploited correctly then calculations can be signficantly accelerated by exploting the underlying C, C++, and Fortran code.


I found myself in a situation where NumPy was readily available but other approaches were not. 


References and Notes
--------------------

NumPy
   - `NumPy <https://numpy.org/>`_

Publications
   - `Lorensen and Cline 1987 Original Marching Cubes Paper <https://dx.doi.org/10.1145/37402.37422>`_
   - `Lorensen's Historical Perspective of Marching Cubes <https://dx.doi.org/10.1109/MCG.2020.2971284>`_

Wikipedia
   - `Wikipedia: Marching Cubes <https://en.wikipedia.org/wiki/Marching_cubes>`_
   - `Wikipedia: Marching Squares <https://en.wikipedia.org/wiki/Marching_squares>`_
   - `Wikipedia: Dual Contouring <https://en.wikipedia.org/wiki/Isosurface#Dual_contouring>`_

Blogs and Tutorials
   - `Boris the Brave: Marching Cubes <https://www.boristhebrave.com/2018/04/15/marching-cubes-3d-tutorial/>`_
   - `Boris the Brave: Dual Contouring <https://www.boristhebrave.com/2018/04/15/dual-contouring-tutorial/>`_
   - `Paul Bourke <http://paulbourke.net/geometry/polygonise/>`_

Other Marching Solutions
   - `scikit-image Introduction <https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html>`_
   - `scikit-image API <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes>`_