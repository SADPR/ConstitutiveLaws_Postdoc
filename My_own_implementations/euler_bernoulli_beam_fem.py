# euler_bernoulli_beam_fem.py
# Minimal 1D Euler–Bernoulli beam (Hermite 2-node element, DOFs = [w, theta] per node)
# - Problem: cantilever beam of length L, EI = E*I
# - Loads: choose tip load P OR uniform load q
# - BCs: w(0)=0, theta(0)=0 (clamped at x=0)
# - Checks: compares deflection to analytical solution for tip load

import numpy as np
import matplotlib.pyplot as plt

# ---------------- User inputs ----------------
L   = 1.0          # beam length [m]
E   = 210e9        # Young's modulus [Pa]
I   = 1.0e-6       # second moment of area [m^4]
nel = 10           # number of finite elements (≥ 2 recommended)

use_tip_load   = True
P   = 1000.0       # tip transverse load at x=L [N] (used if use_tip_load=True)
q   = 0.0          # uniform transverse load [N/m]   (used if use_tip_load=False)

# ---------------- Discretization -------------
nn   = nel + 1                      # number of nodes
xe   = np.linspace(0.0, L, nn)      # node coordinates
Le   = L / nel                      # element length (uniform mesh)

ndof_per_node = 2                   # [w, theta] per node
ndof = ndof_per_node * nn

# DOF ordering: [w0, th0, w1, th1, ..., wN, thN]
def dof_ids(e):
    """Global DOF indices for element e (0-based)."""
    n1 = e
    n2 = e + 1
    return np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1], dtype=int)

# Element stiffness (standard 2-node Euler–Bernoulli)
def beam_ke(EI, Le):
    L2 = Le**2
    return (EI / (Le**3)) * np.array([
        [ 12.0,     6.0*Le,  -12.0,     6.0*Le],
        [  6.0*Le,  4.0*L2,   -6.0*Le,   2.0*L2],
        [-12.0,    -6.0*Le,   12.0,    -6.0*Le],
        [  6.0*Le,  2.0*L2,   -6.0*Le,   4.0*L2],
    ])


# Consistent element load vector for uniform q (downward positive)
# f_e = q * Le/2 * [1, Le/6, 1, -Le/6]^T
def beam_fe_uniform(q, Le):
    return q * Le/2.0 * np.array([1.0, Le/6.0, 1.0, -Le/6.0])

# ---------------- Assembly -------------------
K = np.zeros((ndof, ndof))
F = np.zeros(ndof)

EI = E * I
ke = beam_ke(EI, Le)

for e in range(nel):
    idx = dof_ids(e)
    # assemble stiffness
    K[np.ix_(idx, idx)] += ke
    # assemble consistent distributed load (if used)
    if not use_tip_load and q != 0.0:
        fe = beam_fe_uniform(q, Le)
        F[idx] += fe

# Tip point load at x=L applied at the last node's w DOF
if use_tip_load and P != 0.0:
    F[-2] += P  # last node's 'w' DOF is index -2; last 'theta' is -1

# ---------------- Boundary conditions --------
# Cantilever at x=0: w(0)=0, theta(0)=0
fixed_dofs = np.array([0, 1])  # w0, th0
free_dofs  = np.setdiff1d(np.arange(ndof), fixed_dofs)

# Reduce system
Kff = K[np.ix_(free_dofs, free_dofs)]
Kfc = K[np.ix_(free_dofs, fixed_dofs)]
Ff  = F[free_dofs]

# Prescribed values at fixed DOFs (zero)
Uc = np.zeros_like(fixed_dofs, dtype=float)

# Solve
Uf = np.linalg.solve(Kff, Ff - Kfc @ Uc)

# Reconstruct full displacement vector
U = np.zeros(ndof)
U[free_dofs] = Uf
U[fixed_dofs] = Uc

# Extract nodal deflections and rotations
w = U[0::2]
th = U[1::2]

# ---------------- Post-processing ------------
# Analytical deflection for cantilever with tip load P:
# w(x) = P x^2 (3L - x)/(6 E I)
if use_tip_load:
    x = xe
    w_exact = (P * x**2 * (3.0*L - x)) / (6.0 * EI)
    tip_err = abs(w[-1] - w_exact[-1]) / abs(w_exact[-1]) * 100.0
    print(f"Tip deflection FEM = {w[-1]:.6e} m")
    print(f"Tip deflection exact = {w_exact[-1]:.6e} m")
    print(f"Relative error at tip = {tip_err:.3f} %")
else:
    w_exact = None

# Plot
plt.figure()
plt.plot(xe, w, 'o-', label='FEM (deflection)')
if w_exact is not None:
    plt.plot(xe, w_exact, '--', label='Exact (cantilever, tip load)')
plt.xlabel('x [m]')
plt.ylabel('w(x) [m]')
plt.title('Euler–Bernoulli beam (2-node FEM)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

