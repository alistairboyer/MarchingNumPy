
MarchingCuPy
=============

   MarchingCuPy is a direct conversion of MarchingNumPy to use CuPy instead of NumPy.
   The speedup by using CuPy where available is incredible - although there is a cost associated with inital setup.
   
   The module code is created automatically by adding :attr:`import cupy`
   and replacing :attr:`numpy.` array creation routines with their :attr:`cupy.` equivalents. 

   Substitutions:
      - numpy.asarray --> cupy.asarray
      - numpy.concatenate --> cupy.concatenate
      - numpy.full_like --> cupy.full_like
      - numpy.nonzero --> cupy.nonzero
      - numpy.ones --> cupy.ones
      - numpy.zeros --> cupy.zeros
   
   The majority of operations are methods on array objects so are converted implicitly.
   In other cases the :attr:`cupy.` attribute is an alias for the :attr:`numpy.` attribute.

   The code in MarchingNumPy is idential to that in MarchingCuPy with a few subtle differences in operation.

   Operational differences:
      - cupy does not support the ``where=`` keyword argument in methods so an alternative is used
      - :meth:`cupy.einsum` is exceptionally slow so an alternative is used
   

