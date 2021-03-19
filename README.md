# ME 408 - Spectral Methods in Computational Physics - Final Project

The source files for the final exam. 

## Spectral Navier-Stokes solver

### File manifest

`navier-stokes/ns.py` is the solver.

`navier-stokes/subroutines` contains routines for Fourier transforms and auxiliary quantities (vorticity, pressure).

`navier-stokes/data` contains data generated from simulations at various integration times and mesh sizes.

`navier-stokes/figures` contains figures generated using

`navier-stokes/post.py`

## Bessel's equation eigenvalue calculator

### File manifest

`bessel/cb.py` uses a Chebyshev expansion on the singular problem.

`bessel/cb_mod.py` uses a Chebyshev expansion on the non-singular problem.

`bessel/fd.py` uses second-order finite differences.

`bessel/subroutines` contains the routines for Fourier and Chebyshev collocation derivative matrices.

All code in the respository has been written explicitly for use in ME408.
