#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401

# ==============================
# Config / Material
# ==============================
REL_EPS      = 1e-14
PLANE_STRAIN = True   # we assume LinearElasticPlaneStrain2DLaw
E_MAT        = 1.0e8  # Pa
NU_MAT       = 0.4

# Affine test field u = A X
ALPHA = 1e-3   # ε_xx
BETA  = 2e-3   # ε_yy
S     = 1e-3   # ε_xy = S  (engineering γ_xy = 2S)

# ==============================
# Helpers
# ==============================

def rel_err(a, b, eps=REL_EPS):
    denom = np.maximum(np.abs(b), eps)
    return np.abs(a - b) / denom


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


def assemble_global_B(mp, idx_ux, idx_uy):
    """
    Build global operator B so that eps_all = B @ u,
    where u is packed with the node-based map.
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

            B_loc = build_B_from_DNDX(DNDX)                            # (3,2*nnode)

            for i in range(3):  # 3 rows for this GP
                r = row_base + i
                for a in range(2*nnode):
                    rows.append(r)
                    cols.append(col_ids[a])
                    vals.append(B_loc[i, a])

            gp_meta.append((elem.Id, igauss))
            row_base += 3

    n_rows = row_base
    n_cols = 2*len(idx_ux)

    try:
        from scipy.sparse import coo_matrix
        B_glob = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    except Exception:
        B_glob = np.zeros((n_rows, n_cols))
        for r, c, v in zip(rows, cols, vals):
            B_glob[r, c] += v

    return B_glob, gp_meta


def gather_ip_arrays(mp, variable_vec, voigt_size_expected=3):
    """
    Flatten integration-point vectors (XX, YY, XY) over all elements.
    Returns:
        flat    : 1D array of size 3 * n_gp_total
        gp_meta : list of (elem_id, gp_index)
    """
    flat = []
    gp_meta = []
    process_info = mp.ProcessInfo

    for elem in mp.Elements:
        vals = elem.CalculateOnIntegrationPoints(variable_vec, process_info)
        for igauss, v in enumerate(vals):
            arr = np.array(v, dtype=float)
            print(elem.Id)
            print(variable_vec)
            print(arr)
            if arr.shape[0] < voigt_size_expected:
                raise RuntimeError(f"Voigt size < {voigt_size_expected} for elem {elem.Id}")
            flat.extend(arr[:3])  # XX, YY, XY
            gp_meta.append((elem.Id, igauss))

    return np.array(flat), gp_meta


def build_C_plane_strain(E, nu):
    """
    Plane-strain isotropic elasticity with engineering shear:
    eps_eng = [exx, eyy, gxy], sigma = [sxx, syy, txy].
    """
    lam = E*nu / ((1+nu)*(1-2*nu))
    mu  = E/(2*(1+nu))
    C = np.array([[lam + 2*mu, lam,          0.0],
                  [lam,        lam + 2*mu,   0.0],
                  [0.0,        0.0,          mu]])
    return C


def impose_affine_displacement_and_fix(mp, A, step_index=0):
    """
    u(X,Y) = A @ [X,Y]^T, and fix all displacement DOFs.
    """
    for node in mp.Nodes:
        X, Y = node.X0, node.Y0
        ux = A[0,0]*X + A[0,1]*Y
        uy = A[1,0]*X + A[1,1]*Y

        disp = KM.Array3()
        disp[0] = ux
        disp[1] = uy
        disp[2] = 0.0

        node.SetSolutionStepValue(KM.DISPLACEMENT,   step_index, disp)
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, step_index, ux)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, step_index, uy)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Z, step_index, 0.0)

        node.Fix(KM.DISPLACEMENT_X)
        node.Fix(KM.DISPLACEMENT_Y)
        node.Fix(KM.DISPLACEMENT_Z)


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    # --- Load parameters & analysis stage class (as in your big script) ---
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KM.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name  = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name  = ''.join(x.title() for x in analysis_stage_class_name.split('_'))
    analysis_stage_module      = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class       = getattr(analysis_stage_module, analysis_stage_class_name)

    # --- Create one simulation (single batch-style) ---
    model = KM.Model()
    sim   = analysis_stage_class(model, parameters)

    # We don't want your original batch_strain BCs here, so set to zero:
    if hasattr(sim, "batch_strain"):
        sim.batch_strain = np.zeros(3)

    # Initialize stage (build solver, import model part, etc.)
    sim.Initialize()

    mp = sim._GetSolver().GetComputingModelPart()
    print("[INFO] Model part:", mp.Name)
    print("[INFO] #Nodes:", mp.NumberOfNodes(), "| #Elements:", mp.NumberOfElements())

    # Build global maps and B once
    idx_ux, idx_uy, _ = build_node_global_map(mp)
    B_glob, gp_meta_B = assemble_global_B(mp, idx_ux, idx_uy)
    n_gp_total = len(gp_meta_B)

    # Affine field matrix A from desired engineering strains:
    # eps_eng = [exx, eyy, gxy] = [ALPHA, BETA, 2S]
    # For symmetric gradient: A = [[exx, s],[s, eyy]] with gxy = 2s.
    A_mat = np.array([[ALPHA, S],
                      [S,     BETA]])
    print("[INFO] Affine A used for u = A X:")
    print(A_mat)

    # --- Time step & "solve nothing" ---
    sim.time = sim._AdvanceTime()   # advance to first time
    sim.InitializeSolutionStep()

    # Override nodal displacements with affine field and fix DOFs
    impose_affine_displacement_and_fix(mp, A_mat, step_index=0)

    # Let Kratos "solve" (should see nothing because everything is fixed)
    sim._GetSolver().Predict()
    sim._GetSolver().SolveSolutionStep()
    sim.FinalizeSolutionStep()

    # --- Now postprocess: build u, B u, GL, PK2, compare ---

    # 1) Pack global u and compute eps_B = B u
    u = pack_global_u(mp, idx_ux, idx_uy, step_index=0)
    eps_B = B_glob.dot(u) if hasattr(B_glob, "dot") else (B_glob @ u)
    eps_B_reshaped = eps_B.reshape(-1, 3)

    # 2) Analytical strain target per GP
    target_gp = np.array([ALPHA, BETA, 2*S])  # [εxx, εyy, γxy]
    eps_target = np.tile(target_gp, n_gp_total).reshape(-1, 3)

    err_eps = np.abs(eps_B_reshaped - eps_target)
    print("[B-TEST] max |B u - analytic ε| = {:.3e}".format(err_eps.max()))

    # 3) Build C and compute stresses from B u
    C = build_C_plane_strain(E_MAT, NU_MAT)
    s_from_B = (C @ eps_B_reshaped.T).T          # (n_gp_total, 3)
    s_target_gp = C @ target_gp                  # (3,)
    s_target_all = np.tile(s_target_gp, (n_gp_total, 1))

    err_s_C = np.abs(s_from_B - s_target_all)
    print("[C-TEST] target stress (single GP) =")
    print("  sigma_xx =", s_target_gp[0])
    print("  sigma_yy =", s_target_gp[1])
    print("  tau_xy   =", s_target_gp[2])
    print("[C-TEST] max |C(B u) - C(analytic ε)| = {:.3e}".format(err_s_C.max()))

    # 4) Ask Kratos for GL and PK2 at IPs (after "solve nothing")
    gl_flat,  gp_meta_gl = gather_ip_arrays(mp, KM.GREEN_LAGRANGE_STRAIN_VECTOR, voigt_size_expected=3)
    pk2_flat, gp_meta_pk = gather_ip_arrays(mp, KM.PK2_STRESS_VECTOR,          voigt_size_expected=3)

    if gp_meta_gl != gp_meta_B:
        print("[WARN] gp_meta mismatch between B and GL; comparison is in flat order only.")

    gl_reshaped  = gl_flat.reshape(-1, 3)
    pk2_reshaped = pk2_flat.reshape(-1, 3)

    # 5) Compare GL vs B u and vs analytic strain
    err_GL_vs_B   = np.abs(gl_reshaped - eps_B_reshaped)
    err_GL_vs_tgt = np.abs(gl_reshaped - eps_target)

    print("[GL-TEST] max |GL - (B u)|       = {:.3e}".format(err_GL_vs_B.max()))
    print("[GL-TEST] max |GL - analytic ε|  = {:.3e}".format(err_GL_vs_tgt.max()))

    # 6) Compare PK2 vs C(B u)
    err_PK2_vs_B = np.abs(pk2_reshaped - s_from_B)
    rel_PK2_vs_B = rel_err(pk2_reshaped, s_from_B)

    print("[PK2-TEST] max |PK2 - C(B u)|    = {:.3e}".format(err_PK2_vs_B.max()))
    print("[PK2-TEST] max rel|PK2 - C(B u)| = {:.3e}".format(rel_PK2_vs_B.max()))

    # Optional: print first few GPs
    print("\n[CHECK] First 3 GPs:")
    for i in range(n_gp_total):
        exx_B, eyy_B, gxy_B = eps_B_reshaped[i]
        exx_GL, eyy_GL, gxy_GL = gl_reshaped[i]
        sxx_B, syy_B, txy_B = s_from_B[i]
        sxx_K, syy_K, txy_K = pk2_reshaped[i]
        print(f" GP {i:2d}:")
        print(f"   eps_B   = [{exx_B:.6e}, {eyy_B:.6e}, {gxy_B:.6e}]")
        print(f"   GL_K    = [{exx_GL:.6e}, {eyy_GL:.6e}, {gxy_GL:.6e}]")
        print(f"   sigma_B = [{sxx_B:.6e}, {syy_B:.6e}, {txy_B:.6e}]")
        print(f"   PK2_K   = [{sxx_K:.6e}, {syy_K:.6e}, {txy_K:.6e}]")

    sim.Finalize()
    print("\n[DONE] Affine u = A X test with B, C, GL, PK2 comparisons.")
