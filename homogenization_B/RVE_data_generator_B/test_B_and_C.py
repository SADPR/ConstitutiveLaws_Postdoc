#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import KratosMultiphysics as KM
import importlib
from scipy.sparse import coo_matrix
import time

# ==============================
# Config / Material (global defaults, kept for reference)
# ==============================
REL_EPS      = 1e-14
PLANE_STRAIN = True   # we assume LinearElasticPlaneStrain2DLaw
E_MAT        = 1.0e8  # Pa
NU_MAT       = 0.4

# Affine test field u = A X
ALPHA = 1e-3   # ε_xx
BETA  = 2e-3   # ε_yy
S     = 1e-3   # ε_xy = S  (engineering γ_xy = 2S)

STRUCTURAL_MATERIALS_FILE = "StructuralMaterials.json"

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
    Deterministic map consistent with TensorAdaptor ordering:
    [ux(node0), uy(node0), ux(node1), uy(node1), ...] following mp.Nodes iteration.
    """
    idx_ux, idx_uy = {}, {}
    for k, node in enumerate(mp.Nodes):  # same order that TensorAdaptors use
        idx_ux[node.Id] = 2*k
        idx_uy[node.Id] = 2*k + 1
    n_dof = 2*mp.NumberOfNodes()
    return idx_ux, idx_uy, n_dof


def assemble_global_B(mp, idx_ux, idx_uy):
    """
    Build global operator B so that eps_all = B @ u,
    where u is packed with the node-based map.
    3 rows per GP, columns per (ux,uy) of each node.

    Returns:
        B_glob : csr_matrix (3*n_gp_total, 2*n_nodes)
        gp_meta: list of (elem_id, igauss) in the same order as the GP rows
                 in B_glob (i.e., every 3 consecutive rows belong to one GP).
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
    
    B_glob = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

    return B_glob, gp_meta


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


def build_C_matrices_from_structural_materials(filename=STRUCTURAL_MATERIALS_FILE):
    """
    Kratos-style parsing of StructuralMaterials.json using KM.Parameters.

    Returns:
        C_by_props_id: dict {properties_id (int): C (3x3 np.array)}
    """
    with open(filename, "r") as f:
        mat_params = KM.Parameters(f.read())

    C_by_props_id = {}
    props_list = mat_params["properties"]

    for i in range(props_list.size()):
        props_i = props_list[i]
        pid = props_i["properties_id"].GetInt()

        mat_vars = props_i["Material"]["Variables"]
        E  = mat_vars["YOUNG_MODULUS"].GetDouble()
        nu = mat_vars["POISSON_RATIO"].GetDouble()

        C_by_props_id[pid] = build_C_plane_strain(E, nu)

    print("[INFO] Built C matrices from StructuralMaterials.json:")
    for pid, C in C_by_props_id.items():
        print(f"  properties_id = {pid}")
        print(C)

    return C_by_props_id


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

    # Load constitutive matrices per material (properties_id) from StructuralMaterials.json
    C_by_props_id = build_C_matrices_from_structural_materials(STRUCTURAL_MATERIALS_FILE)

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
    ta_disp = KM.TensorAdaptors.HistoricalVariableTensorAdaptor(
        mp.Nodes, KM.DISPLACEMENT, data_shape=[2]
    )  # only (ux, uy) since this is 2D
    ta_disp.CollectData()

    # ta_disp.data has shape (n_nodes, 2) ordered by node Id
    disp_array = ta_disp.data

    # Flatten to [ux1, uy1, ux2, uy2, ...]
    u = disp_array.reshape(-1)

    # Compute B u
    eps_B = B_glob.dot(u) if hasattr(B_glob, "dot") else (B_glob @ u)
    eps_B_reshaped = eps_B.reshape(-1, 3)

    # 2) Analytical strain target per GP (kinematic, same for all materials)
    target_gp = np.array([ALPHA, BETA, 2*S])  # [εxx, εyy, γxy]
    eps_target = np.tile(target_gp, n_gp_total).reshape(-1, 3)

    err_eps = np.abs(eps_B_reshaped - eps_target)
    print("[B-TEST] max |B u - analytic ε| = {:.3e}".format(err_eps.max()))

    # 3) Compute stresses from B u using C_by_props_id (multi-material)

    s_from_B     = np.zeros_like(eps_B_reshaped)  # (n_gp_total, 3)
    s_target_all = np.zeros_like(eps_B_reshaped)  # (n_gp_total, 3)

    for igp, (elem_id, igauss) in enumerate(gp_meta_B):
        elem = mp.Elements[elem_id]
        pid  = elem.Properties.Id  # should correspond to "properties_id" in StructuralMaterials.json

        C = C_by_props_id[pid]

        # sigma_from_B = C * (eps_B at this GP)
        s_from_B[igp, :] = C @ eps_B_reshaped[igp, :]

        # target stress = C * target_gp (same analytic strain, material-specific C)
        s_target_all[igp, :] = C @ target_gp

    err_s_C = np.abs(s_from_B - s_target_all)
    print("[C-TEST] max |C(B u) - C(analytic ε)| = {:.3e}".format(err_s_C.max()))

    # 4) Get GREEN_LAGRANGE_STRAIN_VECTOR and PK2_STRESS_VECTOR
    #    at Gauss points using GaussPointVariableTensorAdaptor (Elements)

    # --- GREEN_LAGRANGE_STRAIN_VECTOR ---
    gp_gl = KM.TensorAdaptors.GaussPointVariableTensorAdaptor(
        mp.Elements, KM.GREEN_LAGRANGE_STRAIN_VECTOR, mp.ProcessInfo
    )
    gp_gl.Check()
    gp_gl.CollectData()

    gl_gp_data  = gp_gl.data       # shape: (n_elem, n_gp_elem, voigt_size)
    gl_gp_shape = gp_gl.DataShape()  # e.g. [n_gp_elem, voigt_size]
    voigt_size_gl = gl_gp_shape[1]

    # (n_gp_total, 3): keep first three Voigt components (XX, YY, XY)
    gl_reshaped = gl_gp_data.reshape(-1, voigt_size_gl)[:, :3]

    # --- PK2_STRESS_VECTOR ---
    gp_pk2 = KM.TensorAdaptors.GaussPointVariableTensorAdaptor(
        mp.Elements, KM.PK2_STRESS_VECTOR, mp.ProcessInfo
    )
    gp_pk2.Check()
    gp_pk2.CollectData()

    pk2_gp_data  = gp_pk2.data     # shape: (n_elem, n_gp_elem, voigt_size)
    pk2_gp_shape = gp_pk2.DataShape()
    voigt_size_pk2 = pk2_gp_shape[1]

    # (n_gp_total, 3): keep first three Voigt components (XX, YY, XY)
    pk2_reshaped = pk2_gp_data.reshape(-1, voigt_size_pk2)[:, :3]

    # 5) Compare GL vs B u and vs analytic strain
    err_GL_vs_B   = np.abs(gl_reshaped - eps_B_reshaped)
    err_GL_vs_tgt = np.abs(gl_reshaped - eps_target)

    print("[GL-TEST] max |GL - (B u)|       = {:.3e}".format(err_GL_vs_B.max()))
    print("[GL-TEST] max |GL - analytic ε|  = {:.3e}".format(err_GL_vs_tgt.max()))

    # 6) Compare PK2 vs C(B u) (multi-material)
    err_PK2_vs_B = np.abs(pk2_reshaped - s_from_B)
    rel_PK2_vs_B = rel_err(pk2_reshaped, s_from_B)

    print("[PK2-TEST] max |PK2 - C(B u)|    = {:.3e}".format(err_PK2_vs_B.max()))
    print("[PK2-TEST] max rel|PK2 - C(B u)| = {:.3e}".format(rel_PK2_vs_B.max()))

    # Optional: print first few GPs
    print("\n[CHECK] First 3 GPs:")
    for i in range(min(3, n_gp_total)):
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
    print("\n[DONE] Affine u = A X test with B, C (multi-material), GL, PK2 comparisons.")
