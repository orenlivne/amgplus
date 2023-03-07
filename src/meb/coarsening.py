"""MEB coarsening of an eigen-neighborhood."""
import numpy as np
import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive
from scipy.sparse.linalg import eigsh
from scipy.linalg import subspace_angles


def create_eigenproblem(domain_size: float, n: int, relaxer) -> hm.hierarchy.multilevel.Level:
    """Creates fine-level eigenproblem: -Delta*u = lam*u with periodic BC. Written as A*x = lam*B*x."""
    a = hm.linalg.sparse_circulant(np.array([-1, 2, -1], dtype=float), np.array([-1, 0, 1]), n)
    b = hm.linalg.sparse_circulant(np.array([1], dtype=float), np.array([0]), n)
    level = hm.setup.hierarchy.create_finest_level(a, b=b, relaxer=relaxer(a, b=b))

    # Initialize hierarchy to 1-level.
    # 'location' is an array of variable locations at all levels. Used only for interpolation neighbor determination.
    h = domain_size / n
    level.location = np.arange(n) * h
    # Required for 'create_coarsening'.
    # TODO(oren): Consider adding 'domain_size' to Level object.
    level.domain_size = domain_size
    return level


def create_coarsening(level: hm.hierarchy.multilevel.Level, num_tv: int, num_sweeps_tv: int, caliber: int):
    # Generate initial test vectors.
    x = hm.setup.auto_setup.get_test_matrix(level.a, num_sweeps_tv, num_examples=num_tv)

    # Coarsening operator.
    aggregate_size = 2
    num_components = 1
    r_stencil = helmholtz.repetitive.coarsening_repetitive.Coarsener(np.array([[0.5, 0.5]]))
    # Convert to sparse matrix + tile over domain.
    r = r_stencil.tile(level.a.shape[0] // aggregate_size)

    # Interpolation and coarse equations.
    p = hm.setup.auto_setup.create_interpolation(
        x, level.a, r, level.location, level.domain_size, "ls", aggregate_size=aggregate_size,
        num_components=num_components,
        neighborhood="extended" , repetitive=False, target_error=0.1,
        caliber=caliber, fit_scheme="plain", weighted=False)
    return hm.setup.hierarchy.create_coarse_level(level.a, level.b, r, p, p.T)