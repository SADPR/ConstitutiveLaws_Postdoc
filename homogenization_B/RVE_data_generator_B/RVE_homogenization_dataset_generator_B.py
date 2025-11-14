#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp
import numpy as np

# ==============================
# Config
# ==============================
TOL = 1e-9
PLANE_STRAIN = True  # set False for plane stress
SMALL_STRAIN_TEST = True  # we compare GREEN_LAGRANGE ~ small strain (engineering) if True

# Choose ONE (theta, phi) and amplitude
THETA = 25.0  # degrees
PHI   = 40.0  # degrees
MAX_STRETCH = 1.0e-4  # small so GL ≈ small-strain

# ==============================
# Helpers: kinematics & operators
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
        # εyy
        B[1, 2*a + 1] = Ny
        # γxy = ∂u/∂y + ∂v/∂x (engineering shear)
        B[2, 2*a    ] = Ny
        B[2, 2*a + 1] = Nx
    return B

def build_node_global_map(mp):
    """Deterministic map: [ux(1), uy(1), ux(2), uy(2), ...] with nodes sorted by Id."""
    ids = sorted([node.Id for node in mp.Nodes])
    idx_ux, idx_uy = {}, {}
    for k, nid in enumerate(ids):
        idx_ux[nid] = 2*k
        idx_uy[nid] = 2*k + 1
    return idx_ux, idx_uy, 2*len(ids)

def pack_global_u(mp, idx_ux, idx_uy, step_index=0):
    """Pack u according to the deterministic node-based map."""
    u = np.zeros(2*len(idx_ux))
    for node in mp.Nodes:
        uv = node.GetSolutionStepValue(KM.DISPLACEMENT, step_index)
        u[idx_ux[node.Id]] = uv[0]
        u[idx_uy[node.Id]] = uv[1]
    return u

def assemble_global_G(mp, idx_ux, idx_uy):
    """
    Build G so that eps_all = G @ u.
    Three rows per Gauss point, cols are (ux,uy) per node. Stores gp_meta as (elem.Id, igauss).
    """
    try:
        from scipy.sparse import coo_matrix
        use_sparse = True
    except Exception:
        use_sparse = False

    rows, cols, vals, gp_meta = [], [], [], []
    row_base = 0

    for elem in mp.Elements:
        geom = elem.GetGeometry()
        nnode = geom.PointsNumber()

        # cols for this element
        col_ids = []
        for node in geom:
            col_ids.extend([idx_ux[node.Id], idx_uy[node.Id]])

        Ns = np.array(geom.ShapeFunctionsValues())   # (n_gp, nnode)
        n_gp = Ns.shape[0]

        for igauss in range(n_gp):
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))  # (nnode,2) wrt (ξ,η)
            J    = np.array(geom.Jacobian(igauss))                     # (2,2) ∂x/∂ξ
            # more stable than explicit inverse
            DNDX = np.linalg.solve(J.T, DNDe.T).T

            Be = build_B_from_DNDX(DNDX)  # (3, 2*nnode)

            # append block (3 rows)
            rows.extend(row_base + np.repeat([0,1,2], 2*nnode))
            cols.extend(np.tile(col_ids, 3))
            vals.extend(Be.ravel())

            gp_meta.append((elem.Id, igauss))
            row_base += 3

    n_rows = row_base
    n_cols = 2*mp.NumberOfNodes()

    if use_sparse:
        G = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    else:
        G = np.zeros((n_rows, n_cols))
        for r, c, v in zip(rows, cols, vals):
            G[r, c] += v

    return G, gp_meta

def read_linear_elastic_properties(mp):
    """Reads E and nu from the first element's Properties."""
    for elem in mp.Elements:
        props = elem.Properties
        E  = props.GetValue(KM.YOUNG_MODULUS)
        nu = props.GetValue(KM.POISSON_RATIO)
        return float(E), float(nu)
    raise RuntimeError("No elements found to read material properties.")

def build_C(E, nu, plane_strain=True):
    """3x3 elastic matrix in Voigt with engineering shear (γ)."""
    if plane_strain:
        lam = E*nu/((1+nu)*(1-2*nu))
        mu  = E/(2*(1+nu))
        C = np.array([[lam+2*mu, lam,       0.0],
                      [lam,       lam+2*mu, 0.0],
                      [0.0,       0.0,      mu]])
    else:  # plane stress
        fac = E/(1-nu**2)
        C = np.array([[fac,      fac*nu, 0.0],
                      [fac*nu,   fac,    0.0],
                      [0.0,      0.0,    E/(2*(1+nu))]])
    return C

def gather_ip_arrays(mp, variable_vec, voigt_size_expected=3):
    """
    Gathers an array of size 3*n_ip_total matching the same (elem,igauss) order we use in assemble_global_G.
    variable_vec is a Kratos vector variable at IPs (e.g., GREEN_LAGRANGE_STRAIN_VECTOR, PK2_STRESS_VECTOR).
    Returns flat array and (gp_meta) list.
    """
    flat = []
    gp_meta = []
    process_info = mp.ProcessInfo
    for elem in mp.Elements:
        vals = elem.CalculateOnIntegrationPoints(variable_vec, process_info)  # list of vectors (n_gp)
        for igauss, v in enumerate(vals):
            arr = np.array(v, dtype=float)
            if arr.shape[0] < voigt_size_expected:
                raise RuntimeError(f"Voigt size < {voigt_size_expected} for elem {elem.Id}")
            # take first 3 components (XX, YY, XY-engineering) as in 2D
            flat.extend(arr[:3])
            gp_meta.append((elem.Id, igauss))
    return np.array(flat), gp_meta

def homogenize_from_ip(mp, ip_vec_flat, gp_meta, weight_by_area=True):
    """
    ip_vec_flat is stacked as [xx, yy, xy]_gp1, [xx, yy, xy]_gp2, ...
    Returns area-weighted average (3,) using element areas. If multiple GPs per element,
    weights each elem equally across its IPs.
    """
    # Count ips per element
    from collections import defaultdict
    ip_per_elem = defaultdict(int)
    for (eid, ig) in gp_meta:
        ip_per_elem[eid] += 1

    # Accumulate
    num = np.zeros(3)
    den = 0.0
    i = 0
    for elem in mp.Elements:
        area = elem.GetGeometry().Area() if weight_by_area else 1.0
        n_gp = ip_per_elem[elem.Id]
        for ig in range(n_gp):
            num += area * ip_vec_flat[i:i+3]
            den += area
            i += 3
    return num / den

# ==============================
# Your AnalysisStage (unchanged BC logic)
# ==============================
class RVE_homogenization_dataset_generator(analysis_stage.AnalysisStage):
    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        self.batch_strain = np.array([])

    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        order_processes_initialization = self._GetOrderOfProcessesInitialization()
        self._list_of_processes        = self._CreateProcesses("processes", order_processes_initialization)
        deprecated_output_processes    = self._CheckDeprecatedOutputProcesses(self._list_of_processes)
        order_processes_initialization = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_processes_initialization)
        self._list_of_processes.extend(self._list_of_output_processes)
        self._list_of_output_processes.extend(deprecated_output_processes)

    def ApplyBoundaryConditions(self):
        super().ApplyBoundaryConditions()
        Ex, Ey, Exy = self.batch_strain
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            x = node.X0; y = node.Y0
            displ_x = (Ex  * x + Exy * y) * self.time / self.end_time
            displ_y = (Ey  * y + Exy * x) * self.time / self.end_time
            displ_z = 0.0
            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)

    def CalculateHomogenizedStressAndStrain(self):
        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        mp = self._GetSolver().GetComputingModelPart()

        for element in mp.Elements:
            dummy_strain = np.array(element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info))
            break
        n_ips = dummy_strain.shape[0]

        homogenized_stress = np.zeros(3)
        homogenized_strain = np.zeros(3)
        RVE_area = 0.0

        for element in mp.Elements:
            strain = element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            stress = element.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)
            strain_sum = np.sum(np.array(strain)[:, :3], axis=0)  # take first 3 comps
            stress_sum = np.sum(np.array(stress)[:, :3], axis=0)
            area = element.GetGeometry().Area()
            RVE_area += area
            homogenized_strain += area * strain_sum / n_ips
            homogenized_stress += area * stress_sum / n_ips

        homogenized_strain /= RVE_area
        homogenized_stress /= RVE_area
        return homogenized_strain, homogenized_stress

# ==============================
# Main: single (theta,phi) solve + G-comparison
# ==============================
if __name__ == "__main__":
    # Load parameters and stage class
    with open("ProjectParameters.json", 'r') as parameter_file:
        parameters = KM.Parameters(parameter_file.read())
    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name  = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name  = ''.join(x.title() for x in analysis_stage_class_name.split('_'))
    analysis_stage_module      = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class       = getattr(analysis_stage_module, analysis_stage_class_name)

    # Create one simulation
    model = KM.Model()
    sim   = RVE_homogenization_dataset_generator(model, parameters)

    # One (theta, phi) → batch strain (Exx, Eyy, Exy)
    sim.batch_strain = MAX_STRETCH * np.array([
        np.cos(np.radians(PHI)),                               # Exx
        np.sin(np.radians(THETA)) * np.cos(np.radians(PHI)),   # Eyy
        np.sin(np.radians(THETA)) * np.sin(np.radians(PHI)),   # Exy (engineering)
    ])

    print("[INFO] Selected (theta,phi)=", (THETA, PHI))
    print("[INFO] Target batch strain (at end) =", sim.batch_strain)

    # Initialize and solve time steps
    sim.Initialize()

    mp = sim._GetSolver().GetComputingModelPart()
    idx_ux, idx_uy, _ = build_node_global_map(mp)

    # Material for stress-from-G
    try:
        E_mat, nu_mat = read_linear_elastic_properties(mp)
        Cmat = build_C(E_mat, nu_mat, plane_strain=PLANE_STRAIN)
        have_C = True
        print(f"[INFO] Material: E={E_mat:.6g}, nu={nu_mat:.4f} | {'Plane strain' if PLANE_STRAIN else 'Plane stress'}")
    except Exception as e:
        print("[WARN] Could not read (E, nu) from Properties; stress-from-G comparison skipped.")
        have_C = False
        Cmat = None

    step_count = 0
    while sim.KeepAdvancingSolutionLoop():
        sim.time = sim._AdvanceTime()
        sim.InitializeSolutionStep()
        sim._GetSolver().Predict()
        converged = sim._GetSolver().SolveSolutionStep()
        sim.FinalizeSolutionStep()
        step_count += 1

        # --- (A) Gather Kratos IP strains/stresses (GREEN_LAGRANGE, PK2) ---
        gl_flat, gp_meta_gl = gather_ip_arrays(mp, KM.GREEN_LAGRANGE_STRAIN_VECTOR, voigt_size_expected=3)  # [xx,yy,xy]_gp stacked
        pk2_flat, gp_meta_s = gather_ip_arrays(mp, KM.PK2_STRESS_VECTOR, voigt_size_expected=3)

        # --- (B) Build G and compute strains from displacements: eps_G = G u ---
        # (Re-assemble each step to be safe for nonlinear kinematics; cheap anyway)
        G, gp_meta_G = assemble_global_G(mp, idx_ux, idx_uy)
        u = pack_global_u(mp, idx_ux, idx_uy, step_index=0)  # current step
        eps_G = (G.dot(u) if hasattr(G, "dot") else (G @ u))  # length = 3*n_gp_total

        # Sanity: make sure gp orders match
        if gp_meta_gl != gp_meta_G:
            print("[WARN] gp_meta mismatch between Kratos IP order and G assembly order. "
                  "We will still compute max-norms separately but row-wise comparison may not match.")
        n_gp_total = len(gp_meta_G)

        # --- (C) Compare strains (per-GP) ---
        # Kratos GL has [Exx, Eyy, Exy_tens?] – in Kratos vector the 3rd comp is engineering or tensorial?
        # In SMApp GREEN_LAGRANGE_STRAIN_VECTOR returns GL Voigt with engineering shear (2*E_xy) => consistent with our eps_G.
        # If you detect a factor-2 mismatch, adjust here: gl_flat[2::3] *= 2.0 (rarely needed).
        strain_err = np.abs(eps_G - gl_flat)
        print(f"[STEP {step_count:02d}] Strain per-GP: max|G u - GL| = {strain_err.max():.3e}")

        # --- (D) Optionally compute stress from eps_G and compare to Kratos PK2 ---
        if have_C:
            # For TL linear material, PK2 ≈ C : E ; using engineering shear Voigt means s = C * [Exx, Eyy, 2*E_xy]
            # Our eps_G is engineering (≈ 2*E_xy for small).
            s_from_G = np.zeros_like(pk2_flat)
            for i in range(n_gp_total):
                e_vec = eps_G[3*i:3*(i+1)]
                s_from_G[3*i:3*(i+1)] = Cmat @ e_vec
            stress_err = np.abs(s_from_G - pk2_flat)
            print(f"[STEP {step_count:02d}] Stress per-GP: max|C*(G u) - PK2| = {stress_err.max():.3e}")

        # --- (E) Homogenized (volume-averaged) comparison ---
        # From Kratos:
        e_hom_K, s_hom_K = sim.CalculateHomogenizedStressAndStrain()
        # From G route (area-weighted):
        e_hom_G = homogenize_from_ip(mp, eps_G, gp_meta_G, weight_by_area=True)
        if have_C:
            s_hom_G = Cmat @ e_hom_G
        else:
            s_hom_G = None

        print(f"[STEP {step_count:02d}] Homog strain: |G-avg - Kratos|_max = {np.abs(e_hom_G - e_hom_K).max():.3e}")
        if have_C:
            print(f"[STEP {step_count:02d}] Homog stress: |C(G-avg) - Kratos|_max = {np.abs(s_hom_G - s_hom_K).max():.3e}")

    sim.Finalize()

    print("\n[DONE] Single (theta,phi) run with per-GP and homogenized comparisons finished.")
