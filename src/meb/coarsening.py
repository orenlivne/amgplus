"""MEB coarsening of an eigen-neighborhood."""
import numpy as np
import helmholtz as hm
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

    return level
