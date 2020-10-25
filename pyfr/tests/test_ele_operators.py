import numpy as np

from pyfr.inifile import Inifile
from pyfr.shapes import HexShape
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.readers.native import NativeReader
from pyfr.solvers.base import BaseSystem


def test_operators(order, cfg, filename):
    """
    Tests the interpolation (M0), gradient (M4) and
    and divergence (M1) operators in a mesh given by
    filename
    The parameter order should be integer and
    it is used to assess the precision of the
    operators using a polynomial of degree order
    cfg should be an instance of Inifile which
    holds the flux and solution points information
    of the elements of the mesh given by filename
    """

    # System and elements classes
    systemscls = subclass_where(
        BaseSystem, name=cfg.get('solver', 'system')
    )

    # Read the mesh
    meshin = NativeReader(filename)

    # For each element type in the mesh
    for eletype in meshin.keys():
        # skip non-elements keys
        eletypes = ['tri', 'quad', 'hex', 'tet', "pri", "pry"]
        found = [ele in eletype for ele in eletypes]
        if not any(found):
            continue

        # Obtain the nodes of the elements
        mesh = meshin[eletype]

        # Number of dimensions
        ndim = mesh.shape[-1]

        # Get the shape class
        name_ele = eletype.split('_')[1]
        basiscls = subclass_where(BaseShape, name=name_ele)

        # Elements classes
        elementscls = systemscls.elementscls
        eles = elementscls(basiscls, mesh, cfg)

        # Work in double precision
        dtype = np.float64

        # Position of the solution points
        solptspos = eles.ploc_at_np('upts').astype(dtype).swapaxes(0, 1)

        # Number of solution points
        nupts = eles.nupts

        # Define the error function used to test the operators
        # In this case f = (x' + y' + z')^{exponent}
        # Where exponent = order
        # and x_i' = x_i + displacement_i
        # WARNING: we might want to change the basis of
        # the f function
        # The displacement is added to avoid
        # close-to-zero values of f as we are computing
        # relative errors
        exponent = order
        # Compute the displacement to avoid (x, y, z) close to (0, 0, 0)
        displacement = 1.0 - np.array([pts.min() for pts in solptspos])

        def monomial(pos):
            return np.sum(pos, axis=0)

        def function(pos, exponent):
            # Add the displacement
            displ_pos = pos + displacement[:, np.newaxis, np.newaxis]
            return monomial(displ_pos)**exponent

        # Compute the error function at the solution points
        f = function(solptspos, exponent)

        # Test M0 matrix
        # (interpolation to the flux points)

        # Obtain the flux point positions
        fptspos = eles.ploc_at_np('fpts').swapaxes(0, 1)

        # Compute the analytical function value at the fpts
        f_at_fpts = function(fptspos, exponent)

        # Get the m0 operator
        m0 = eles.basis.m0.astype(dtype)

        # Interpolation from solution points to flux points
        f_at_fptsnum = m0 @ f

        # Compute the relative error
        print(eletype, "maximum interpolation relative error",
              np.max(np.abs(f_at_fpts - f_at_fptsnum)/np.abs(f_at_fpts)))

        # Test the gradient operator M4
        # \nabla u = J^{-T} \nabla_\xi u
        # WARNING: we suppose that there is no jump of f
        # at the flux points. Therefore, we neglect
        # the influence of the M6 operator
        # This is only true if all monomials of f
        # are in the polynomial basis of the element

        # smat represents |J|*J^{-T} at solution points
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

        # rcpdjac is |J|^{-1} at solution points
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator M4
        gradop = eles.basis.m4.astype(dtype)

        # Compute the gradient at upts in the reference frame
        gradsoln = gradop @ f
        # gradsoln \in [ndof*3, nCells]
        gradsoln = gradsoln.reshape(ndim, nupts, 1, -1)

        # Multiply by J^{-T} = |J|^{-1}*|J|*J^{-T} = rcpdjac*smat
        gradsolnum = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                               dtype=dtype, casting='same_kind')

        # Compute the analytical gradient at the solution points
        gradsol = exponent*function(solptspos, exponent - 1)
        gradsol = np.array([gradsol for i in range(ndim)])

        # Compute the error
        print(eletype, "maximum gradient relative error ",
              np.max(np.abs(gradsol - gradsolnum[0])/np.abs(gradsol)))

        # Test the Divergence Operator M1
        # \nabla \cdot \vec{f} = |J|^{-1} \nabla_\xi \cdot |J|*J^{-T} f
        # WARNING: we suppose that there is no jump of \vec{f}
        # at the flux points. Therefore, we neglect
        # the influence of the M3 operator
        # This is only true if all monomials of f
        # are contained in the polynomial basis of the element

        # compute the flux
        # this flux is representative of the
        # flux of the linear advection equation
        flux = np.array([f for i in range(ndim)])

        # M1 operator
        m1 = eles.basis.m1.astype(dtype)

        # compute analytical divergence
        divflux = ndim*exponent*function(solptspos, exponent - 1)

        # multiply the flux by |J|*J^{-1}
        # watch out with the indices of the transpose
        # if you want |J|*J^{-T} use 2, 0 instead of 0, 2
        smat = eles.smat_at_np('upts').astype(dtype).transpose(0, 2, 1, 3)

        # Evaluate the transformed flux of the solution
        fluxtrans = np.einsum('ijkl,jkl->ikl', smat, flux,
                              dtype=dtype, casting='same_kind')

        # reshape the flux to apply m0 to it
        fluxtrans = fluxtrans.reshape(ndim*nupts, -1)

        # compute the divergence without taking into account
        # the end points contribution, as the Riemann flux
        # is equal to the extrapolated points flux value
        # when using analytical interpolation
        divfluxnum = m1 @ fluxtrans

        # multiply by the inverse jacobian
        divfluxnum *= rcpdjac

        # Compute error
        print(eletype, "maximum divergence operator relative error ",
              np.max(np.abs(divfluxnum - divflux)/np.abs(divflux)))


# Read cfg with default flux and solution points
cfg = Inifile.load("defaultpts.cfg")

# Mesh filename
filename = "mesh.pyfrm"

# Orders to be tested
orders = range(2, 5)

# For each order
for order in orders:
    # Update the order in the cfg file
    cfg.set('solver', 'order', str(order))

    # Test the operators in the mesh
    print("Testing operators order ", order)
    test_operators(order, cfg, filename)

    print("\n")
