#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import KratosMultiphysics as KM
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers

try:
    from scipy.sparse import coo_matrix, csr_matrix
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ==============================
# Material (for analytic C)
# ==============================
E_MAT  = 1.0e8   # Pa
NU_MAT = 0.4     # -

# ==============================
# Helpers
# ==============================

def build_B_from_DNDX(DNDX):
    """
    DNDX: (nnode, 2) with columns [dN/dx, dN/dy] in GLOBAL coords.
    Returns B (3, 2*nnode) for 2D small strain with ENGINEERING shear (γxy).
    Voigt order: [εxx, εyy, γxy].
    """
    nnode = DNDX.shape[0]
    B = np.zeros((3, 2*nnode))
    for a in range(nnode):
        Nx, Ny = DNDX[a, 0], DNDX[a, 1]
        # εxx
        B[0, 2*a    ] = Nx
        B[0, 2*a + 1] = 0.0
        # εyy
        B[1, 2*a    ] = 0.0
        B[1, 2*a + 1] = Ny
        # γxy (engineering)
        B[2, 2*a    ] = Ny
        B[2, 2*a + 1] = Nx
    return B


def set_affine_nodal_displacements(mp, A, step_index=0):
    """Impose u_I = A * [X_I, Y_I]^T on the given solution step."""
    for node in mp.Nodes:
        X, Y = node.X0, node.Y0
        ux = A[0,0]*X + A[0,1]*Y
        uy = A[1,0]*X + A[1,1]*Y
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, step_index, ux)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, step_index, uy)
        # also set the vector, for completeness / CLs that read the vector
        disp = KM.Array3()
        disp[0] = ux
        disp[1] = uy
        disp[2] = 0.0
        node.SetSolutionStepValue(KM.DISPLACEMENT, step_index, disp)


def build_node_global_map(mp):
    """
    Deterministic map: [ux(node1), uy(node1), ux(node2), uy(node2), ...]
    Returns: idx_ux[node.Id] -> pos, idx_uy[node.Id] -> pos, n_dof
    """
    ids = sorted(node.Id for node in mp.Nodes)
    idx_ux, idx_uy = {}, {}
    for k, nid in enumerate(ids):
        idx_ux[nid] = 2*k
        idx_uy[nid] = 2*k + 1
    n_dof = 2*len(ids)
    return idx_ux, idx_uy, n_dof


def pack_global_u(mp, idx_ux, idx_uy, step_index=0):
    """Pack u according to the deterministic node-based map."""
    n_dof = 2*len(idx_ux)
    u = np.zeros(n_dof)
    for node in mp.Nodes:
        iux = idx_ux[node.Id]
        iuy = idx_uy[node.Id]
        ux = node.GetSolutionStepValue(KM.DISPLACEMENT_X, step_index)
        uy = node.GetSolutionStepValue(KM.DISPLACEMENT_Y, step_index)
        u[iux] = ux
        u[iuy] = uy
    return u


def assemble_global_G(mp, idx_ux, idx_uy):
    """
    Build G so that eps_all = G @ u where u is packed with the node-based map.
    3 rows per GP, columns per (ux,uy) of each node.
    """
    rows, cols, vals = [], [], []
    gp_meta = []
    row_base = 0

    for elem in mp.Elements:
        geom = elem.GetGeometry()
        nnode = geom.PointsNumber()

        # local dof column indices for this element (using our deterministic map)
        col_ids = []
        for node in geom:
            col_ids.append(idx_ux[node.Id])
            col_ids.append(idx_uy[node.Id])

        Ns = np.array(geom.ShapeFunctionsValues())   # (n_gp, nnode)
        n_gp = Ns.shape[0]

        for igauss in range(n_gp):
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))  # (nnode,2)
            J    = np.array(geom.Jacobian(igauss))                     # (2,2)
            DNDX = DNDe @ np.linalg.inv(J)                             # (nnode,2) global

            B = build_B_from_DNDX(DNDX)                                # (3,2*nnode)

            # insert triplets
            for i in range(3):              # 3 rows for this GP
                r = row_base + i
                for a in range(2*nnode):
                    rows.append(r)
                    cols.append(col_ids[a])
                    vals.append(B[i, a])

            gp_meta.append((elem.Id, igauss))
            row_base += 3

    n_rows = row_base
    n_cols = 2*len(idx_ux)

    try:
        from scipy.sparse import coo_matrix
        G = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    except Exception:
        G = np.zeros((n_rows, n_cols))
        for r, c, v in zip(rows, cols, vals):
            G[r, c] += v

    return G, gp_meta


def global_affine_target(A, n_gp_total):
    """
    Build the vector of theoretical engineering strains repeated for all GPs:
    target_gp = [A11, A22, A12 + A21]
    Returns eps_target of size 3*n_gp_total.
    """
    target_gp = np.array([A[0,0], A[1,1], A[0,1] + A[1,0]])  # [εxx, εyy, γxy]
    return np.tile(target_gp, n_gp_total)


def build_C_plane_strain(E, nu):
    """
    Plane-strain isotropic elasticity with engineering shear:
    eps_eng = [exx, eyy, gxy], sigma = [sxx, syy, txy]
    """
    lam = E*nu / ((1+nu)*(1-2*nu))
    mu  = E/(2*(1+nu))
    C = np.array([[lam + 2*mu, lam,          0.0],
                  [lam,        lam + 2*mu,   0.0],
                  [0.0,        0.0,          mu]])
    return C


# ==============================
# Main
# ==============================

def main():
    # 1) Read ProjectParameters.json
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())

    # 2) Build Model and Solver (no full analysis stage needed)
    model = KM.Model()
    solver = structural_solvers.CreateSolver(model, parameters)
    solver.AddVariables()
    solver.ImportModelPart()
    solver.PrepareModelPart()
    solver.AddDofs()

    # 3) Get the computing model part
    mp = solver.GetComputingModelPart()
    print("[Info] Model part:", mp.Name)
    print("[Info] #Nodes:", mp.NumberOfNodes(), "| #Elements:", mp.NumberOfElements())

    # 4) Choose an affine field A and impose nodal u = A X
    alpha, beta, s = 1e-3, 2e-3, 1e-3
    A = np.array([[alpha, s],
                  [s,     beta]])

    print("[Info] Affine A =")
    print(A)

    set_affine_nodal_displacements(mp, A, step_index=0)

    # 5) Build deterministic map and pack u
    idx_ux, idx_uy, n_dof = build_node_global_map(mp)
    u = pack_global_u(mp, idx_ux, idx_uy, step_index=0)

    # 6) Assemble G with the SAME map and compute strains
    G, gp_meta = assemble_global_G(mp, idx_ux, idx_uy)
    eps_all = G.dot(u) if hasattr(G, "dot") else (G @ u)

    n_gp_total = len(gp_meta)

    # theoretical target per GP (engineering strain)
    target_gp = np.array([A[0,0], A[1,1], A[0,1] + A[1,0]])  # [α, β, 2s]
    eps_target = np.tile(target_gp, n_gp_total)

    # 7) Kinematic check: G u vs analytical strain
    err_eps = np.abs(eps_all - eps_target)
    print("[GLOBAL G-TEST] max |Δε| = {:.3e}".format(err_eps.max()))
    print("[GLOBAL G-TEST] target_gp =", target_gp)

    # 8) Constitutive: build C (plane strain) and compute stresses
    C = build_C_plane_strain(E_MAT, NU_MAT)
    print("[Info] C (plane strain, engineering shear) =")
    print(C)

    # (a) Stress from G*u at each GP
    eps_all_reshaped = eps_all.reshape(-1, 3)       # (n_gp_total, 3)
    s_from_G = (C @ eps_all_reshaped.T).T           # (n_gp_total, 3)
    # (b) Analytic target stress (same at all GPs)
    s_target_gp = C @ target_gp                     # (3,)
    s_target_all = np.tile(s_target_gp, (n_gp_total, 1))

    # 9) Stress check
    err_s = np.abs(s_from_G - s_target_all)
    print("[GLOBAL C-TEST] target stress (single GP) =")
    print("  sigma_xx =", s_target_gp[0])
    print("  sigma_yy =", s_target_gp[1])
    print("  tau_xy   =", s_target_gp[2])
    print("[GLOBAL C-TEST] max |Δσ| (C(G u) vs C(analytic eps)) = {:.3e}".format(err_s.max()))

    # Optional: print first few GPs to see everything
    print("\n[CHECK] First 3 GPs:")
    for i in range(n_gp_total):
        exx, eyy, gxy = eps_all_reshaped[i]
        sxx, syy, txy = s_from_G[i]
        print(f" GP {i:2d}: eps = [{exx:.6e}, {eyy:.6e}, {gxy:.6e}]"
              f" | sigma = [{sxx:.6e}, {syy:.6e}, {txy:.6e}]")

if __name__ == "__main__":
    main()
