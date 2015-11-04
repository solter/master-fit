#!/usr/bin/python3.4
"""
This module contains a class which can perform polynomial interpolations
for user defined dimensions, degrees, and grids.
It uses monomials in N-dimensions as the basis.
It also contains the Error classes it can raise.
"""

import numpy as np
import numpy.linalg as la
import itertools as it
from sys import float_info as fi

class PolyInt:
    """
    A class which can create polynomial
    interpolations for an arbitrary set of
    grid points in an arbitrary dimension
    """


    def __init__(self,
            grid=None, regular=False, deg=None, local=True,
            x0=None, f=None, tol=fi.epsilon):
        """
        Initializes a polyInt object
        ============================
        It can be initialized with the following variables:
        grid -- a list of n-dimensional points
        regular -- defines how the grid is defined. See set_grid for details
        deg -- the degree of the polynomial basis to fit
        local -- (bool) if true, polynomial fits will be
            calculated locally to the desired output.
            otherwise will perform a least squares fitting
            of the polynomial to the entire grid (default: True)
        x0 -- a point of interest. defines the 'local' environment
        f -- known values of the function. must be defined such that
            f(x) = f[grid.index(x)]
        tol -- tolerance for least squares fitting. (default: machine epsilon)
        """
        #the grid of points. Will be populated by set_grid method
        self._grid = None
        #list of grid indices to transform f
        self._index_list = None
        #degree of fit
        self._deg = deg
        #polynomial basis powers
        self._polybasis = None
        #whether to fit locally
        self._local = local
        #the zero point for the basis functions.
        #This translation shortens needed radius of convergence.
        self._center = None
        #the point defining locality
        self._x0 = x0
        #the known values
        self._func_vals = f
        #whether it uses a regular grid
        self._regular = regular
        #the tolerance below which svd's are dismissed
        self._tol = tol
        #SVD info
        self._decomp = None
        #the indices of disance
        self._dist_idx = None
        #polynomial coefficients
        self._coef = None

        self._gen_poly_basis()
        self.set_grid(grid, self._regular)
        self.set_func_vals(self._func_vals)

    def set_deg(self, deg):
        """
        Sets the degree of the interpolation
        =====================
        inputs:
        deg -- the degree of polynomials to use for interpolation
        """
        self._deg = deg
        self._gen_poly_basis()

    def _gen_svd(self):
        """
        Generates the svd.
        This allows for a trivial least squares fit calculation
        """
        #check to make sure all the information is available
        if (self._grid is None or
                self._polybasis is None or
                (self._local and self._x0 is None)):
            return

        if self._local:
            #get indices of values closest to x0
            dist = la.norm(self._grid - self._x0, axis=1)
            tmp = np.argsort(dist)
            if tmp == self._dist_idx:
                #This indicates x0 is near the previous x0 used to
                #compute the svd, so we don't need to bother recalculating it
                return
            else:
                #center the grid on the new value
                if self._center is not None:
                    self._grid = self._grid + (self._center - self._x0)
                else:
                    self._grid = self._grid - self._x0

                self._center = self._x0

            self._dist_idx = tmp

            #if minimal, form M only the closest $dof points
            basis_matrix = [
                [
                    self._get_monomial(self._dist_idx[i], j)
                    for j in range(len(self._polybasis))
                    ]
                for i in range(len(self._polybasis))
                ]
        else:
            #center the grid
            mins = [min(self._grid[:, i]) for i in range(self._grid.shape[1])]
            maxs = [max(self._grid[:, i]) for i in range(self._grid.shape[1])]
            #the mean of each corresponding element
            mins = np.array(mins)
            maxs = np.array(maxs)
            tmp = .5*(mins + maxs)
            if (self._center == tmp and
                    self._center != self._x0 and
                    self._decomp is not None):
                #the grid has not changed since, so don't bother computing it
                return
            self._center = tmp
            self._grid = self._grid - self._center

            #if not minimal form M using all the grid and b points
            basis_matrix = [
                [
                    self._get_monomial(self._grid[i], j)
                    for j in range(len(self._polybasis))
                    ]
                for i in range(self._grid.shape[0])
                ]

        basis_matrix = np.matrix(basis_matrix)
        U, S, V = la.svd(basis_matrix, full_matrices=0) # pylint: disable=invalid-name
        #remove small components that don't contribute much to fit
        S = S[S > self._tol] # pylint: disable=invalid-name
        #invert
        self._decomp['UH'] = U[:, 0:len(S)].getH()
        self._decomp['invS'] = [1./x for x in S]
        self._decomp['VH'] = V[0:len(S), :].getH()
        #can now generate solutions via VH*invS*UH*f

        self._compute_coef()
        return

    def _compute_coef(self):
        """
        Computes the coefficients of the monomials
        for the given grid and function values
        """
        if self._func_vals is None or self._decomp is None:
            return

        self._coef = self._decomp['UH'].dot(self._func_vals)
        self._coef = np.diag(self._decomp['invS']).dot(self._coef)
        self._coef = self._decomp['VH'].dot(self._coef)
        return

    def set_grid(self, grid, regular=False):
        """
        define the grid to interpolate over

        inputs:
        grid -- a grid of N-dimensional points. Can be specified in 2 ways:
            if regular is False:
                grid is a list of points
            if regular is True:
                grid contains N elements. Each element specifies the list of
                values for the i^th dimension.
                E.G.
                grid[0] = [1,2]
                grid[1] = [3,4]
                points = [(1,3),(1,4),(2,3),(2,4)]
        regular -- defines how the grid is defined. See above
        """
        if grid is None:
            return

        tmp_index_list = None
        if regular:
            tmp_index_list = [
                list(
                    range(len(grid[i]))
                ) for i in range(len(grid))
                ]
            tmp_index_list = list(it.product(*tmp_index_list)) # pylint: disable=star-args
            grid = list(it.product(*grid)) # pylint: disable=star-args

        tmp = np.array(grid)
        if tmp.shape[0] < tmp.shape[1]:
            tmp = np.transpose(tmp)
        #make sure grid contains enough points to provide desired fit level
        if (self._polybasis is not None and
                len(self._polybasis) > self._grid.shape[0]):
            raise UnderdeterminedSystemError("Insufficient grid points. j")

        self._grid = tmp
        self._index_list = tmp_index_list

        self._gen_svd()

        return

    def _gen_poly_basis(self):
        """
        Generates the monomial power sets
        """
        if self._deg is None:
            return

        self._polybasis = []
        for powseq in it.product(range(self._deg + 1), repeat=len(self._x0)):
            if sum(powseq) <= self._deg:
                self._polybasis.append(np.array(powseq))

        #make sure grid contains enough points to provide desired fit level
        if (self._grid is not None and
                len(self._polybasis) > self._grid.shape[0]):
            raise UnderdeterminedSystemError()

        self._gen_svd()

        return

    def set_tol(self, tol):
        """
        redefine the tolerance

        input:
        tol -- new choice for the least squares tolerance
        """
        self._tol = tol
        self._gen_svd()

    def _get_monomial(self, loc, idx):
        """
        Computes the value of the monomial[idx] at loc
        """
        loc = np.array(loc)
        #product of each element raised to its respective power
        return np.prod(loc**self._polybasis(idx))

    def set_func_vals(self, func_vals):
        """
        (Re)Define the list of known values.
        Must be single valued.

        This class is designed such that updating f to a different
        value should be computationally cheap

        inputs:
        func_vals -- known values of the function. must be defined such that
            (free form grid)
            f(x) = f[grid.index(x)]
            (regular grid)
            f(x) = f[grid[0].index(x[0]), grid[1].index(x[1]), ...]
        """
        if self._regular:
            self._func_vals = [func_vals[loc] for loc in self._index_list]
        else:
            self._func_vals = func_vals

        self._compute_coef()

    def __call__(self, x0=None, func_vals=None, force_nonlocal_x0=False): # pylint: disable=invalid-name
        """
        uses the interpolated function to calculate the value at x0

        the inputs are optional if they have already been given to
        the function
        inputs:
        x0 -- the point of interest
        f -- the known point values.
            Equivalent to calling set_func_vals(f) then this function
        force_nonlocal_x0 -- (bool) forces the use of the current polynomial fit
            regardless of whether the class uses local fit
            CHANGING THIS TO TRUE COULD LEAD TO VERY POOR FITTING AT x0
        """
        if (self._local and
                x0 is not None and
                x0 != self._x0 and
                not force_nonlocal_x0):
            #update the svd
            self._x0 = x0
            self._gen_svd()
        elif x0 != self._x0:
            self._x0 = x0
        if self._local and x0 is None and self._x0 is None:
            raise MissingArgumentError(
                "Must define x0 to get an interpolated value"
                )
        if func_vals is not None:
            self.set_func_vals(func_vals)
        if func_vals is None and self._coef is None:
            raise MissingArgumentError(
                "Must define f to get an interpolated value"
                )

        #compute the result
        basisfunc = [
            self._get_monomial(self._x0 - self._center, self._polybasis[j])
            for j in range(len(self._polybasis))
            ]
        return self._coef.dot(basisfunc)


class OutOfBoundsError(Exception):
    """
    Indicates desired location is outside of bounding supercube of
    known points
    """
    pass
class DimensionMismatchError(Exception):
    """
    Indicates dimension of desired location does not match
    dimension of grid points
    """
    pass
class UnderdeterminedSystemError(Exception):
    """
    Indicates that there are too few grid points given
    to fit the polynomial degree asked for
    """
    pass
class MissingArgumentError(Exception):
    """
    Indicates that there was insufficient information
    to compute an interpolated function value
    """
    pass
