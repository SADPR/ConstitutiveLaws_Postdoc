#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp  # noqa: F401
from scipy.sparse import coo_matrix
import os
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

"""
Minimal RVE small-strain homogenization with our own solver.

Kratos is only used to:
  - Read model, geometry, Properties, BC processes (from ProjectParameters.json)
  - Mark which nodes are fixed and prescribe DISPLACEMENT_X/Y (via ApplyBoundaryConditions)

We do:
  - Build B_glob and gp_meta_B (for ε = B u)
  - Build C matrices from StructuralMaterials.json
  - Assemble global K element-wise: K = sum_e sum_gp B_e^T (w_e C_e) B_e
  - For each time step:
      * Let Kratos processes apply BCs (to set fixed node displacements)
      * Extract Dirichlet BCs (dof indices + values)
      * Solve K u = 0 with those Dirichlet BCs
      * ε = B u, σ = C ε
      * Homogenize strain/stress from (ε, σ)

No GL/PK2, no Kratos structural solver, no Kratos-based homogenization.
"""

# ==============================
# Global config / files
# ==============================
REL_EPS = 1e-14
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
        # γxy (engineering γxy = 2εxy)
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

        col_ids = []
        for node in geom:
            col_ids.append(idx_ux[node.Id])
            col_ids.append(idx_uy[node.Id])

        Ns = np.array(geom.ShapeFunctionsValues())   # (n_gp, nnode)
        n_gp = Ns.shape[0]

        for igauss in range(n_gp):
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))  # (nnode,2)
            J    = np.array(geom.Jacobian(igauss))                     # (2,2)
            DNDX = DNDe @ np.linalg.inv(J)                             # (nnode,2)

            B_loc = build_B_from_DNDX(DNDX)                            # (3,2*nnode)

            for i in range(3):
                r = row_base + i
                for a in range(2*nnode):
                    rows.append(r)
                    cols.append(col_ids[a])
                    vals.append(B_loc[i, a])

            gp_meta.append((elem.Id, igauss))
            row_base += 3

    n_rows = row_base
    n_cols = 2 * len(idx_ux)

    B_glob = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    return B_glob, gp_meta


def build_C_plane_strain(E, nu):
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


def assemble_global_K_elemental(mp, idx_ux, idx_uy, C_by_props_id):
    """
    Assemble global stiffness K using standard elemental assembly:

        K = sum_e sum_gp B_e^T (w_e C_e) B_e

    where w_e = Area_e / n_ips_e.
    """
    n_dof = 2 * mp.NumberOfNodes()
    rows, cols, vals = [], [], []

    for elem in mp.Elements:
        geom = elem.GetGeometry()
        nnode = geom.PointsNumber()

        col_ids = []
        for node in geom:
            col_ids.append(idx_ux[node.Id])
            col_ids.append(idx_uy[node.Id])

        Ns = np.array(geom.ShapeFunctionsValues())   # (n_gp, nnode)
        n_gp = Ns.shape[0]
        area_e = geom.Area()
        pid = elem.Properties.Id
        C   = C_by_props_id[pid]                     # (3x3)

        Ke = np.zeros((2*nnode, 2*nnode))

        for igauss in range(n_gp):
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))  # (nnode,2)
            J    = np.array(geom.Jacobian(igauss))                     # (2,2)
            DNDX = DNDe @ np.linalg.inv(J)

            B_loc = build_B_from_DNDX(DNDX)                            # (3, 2*nnode)

            w = area_e / n_gp
            M = w * C                                                  # (3x3)
            Ke += B_loc.T @ M @ B_loc

        # scatter Ke
        for a_local, I in enumerate(col_ids):
            for b_local, J in enumerate(col_ids):
                val = Ke[a_local, b_local]
                if abs(val) > 0.0:
                    rows.append(I)
                    cols.append(J)
                    vals.append(val)

    K_glob = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).toarray()
    return K_glob


def extract_dirichlet_bcs(mp, idx_ux, idx_uy, step_index=0):
    """
    Just read which nodes are fixed and what value they want at this step.
    This assumes processes/ApplyBoundaryConditions have already set:
      - FIXED flags
      - DISPLACEMENT_X/Y values
    """
    dofs = []
    vals = []

    for node in mp.Nodes:
        if node.IsFixed(KM.DISPLACEMENT_X):
            dofs.append(idx_ux[node.Id])
            vals.append(node.GetSolutionStepValue(KM.DISPLACEMENT_X, step_index))
        if node.IsFixed(KM.DISPLACEMENT_Y):
            dofs.append(idx_uy[node.Id])
            vals.append(node.GetSolutionStepValue(KM.DISPLACEMENT_Y, step_index))

    if not dofs:
        return np.zeros(0, dtype=int), np.zeros(0)
    return np.array(dofs, dtype=int), np.array(vals, dtype=float)

def prepare_dirichlet_solver_state(K, mp, idx_ux, idx_uy, step_index=0):
    """
    Precompute everything that does NOT change with time:
      - dirichlet_dofs (which DOFs are fixed)
      - free_dofs
      - K_ff, K_fc
      - Cholesky factorization of K_ff

    We call this ONCE per batch, after BC pattern is established.
    """
    n_dof = 2 * mp.NumberOfNodes()

    # Get which DOFs are constrained (we ignore values here)
    dirichlet_dofs, _ = extract_dirichlet_bcs(mp, idx_ux, idx_uy, step_index)
    dirichlet_dofs = np.unique(dirichlet_dofs)  # just in case

    all_dofs = np.arange(n_dof, dtype=int)
    mask = np.ones(n_dof, dtype=bool)
    mask[dirichlet_dofs] = False
    free_dofs = all_dofs[mask]

    # Build sub-blocks ONCE
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fc = K[np.ix_(free_dofs, dirichlet_dofs)]

    # Factorize K_ff once (symmetric positive-definite → Cholesky)
    # cho_factor returns (L, lower_flag)
    K_ff_fact = cho_factor(K_ff, lower=True, overwrite_a=False, check_finite=False)

    solver_state = {
        "n_dof": n_dof,
        "dirichlet_dofs": dirichlet_dofs,
        "free_dofs": free_dofs,
        "K_ff_fact": K_ff_fact,
        "K_fc": K_fc,
    }
    return solver_state

def solve_linear_system_with_dirichlet(K, mp, idx_ux, idx_uy, step_index=0):
    """
    Solve K u = 0 with Dirichlet BCs taken from Kratos.
    We do NOT need to write u back to Kratos if we don't want to.
    """
    n_dof = 2 * mp.NumberOfNodes()
    f = np.zeros(n_dof)

    dirichlet_dofs, dirichlet_vals = extract_dirichlet_bcs(mp, idx_ux, idx_uy, step_index)

    # Global solution
    u = np.zeros(n_dof)
    if dirichlet_dofs.size > 0:
        u[dirichlet_dofs] = dirichlet_vals

    all_dofs = np.arange(n_dof, dtype=int)
    mask = np.ones(n_dof, dtype=bool)
    mask[dirichlet_dofs] = False
    free_dofs = all_dofs[mask]

    res_norm_free = 0.0

    if free_dofs.size > 0:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        K_fc = K[np.ix_(free_dofs, dirichlet_dofs)]
        rhs = - K_fc @ u[dirichlet_dofs]

        u_f = np.linalg.solve(K_ff, rhs)
        res = K_ff @ u_f - rhs
        res_norm_free = float(np.linalg.norm(res))

        u[free_dofs] = u_f

    return u, res_norm_free

def solve_linear_system_with_dirichlet_fast(solver_state, mp, idx_ux, idx_uy, step_index=0):
    """
    Fast solve using precomputed solver_state:
      - dirichlet_dofs, free_dofs
      - K_ff_factorization, K_fc

    Only Dirichlet *values* are allowed to change per step.
    """
    n_dof          = solver_state["n_dof"]
    dirichlet_dofs = solver_state["dirichlet_dofs"]
    free_dofs      = solver_state["free_dofs"]
    K_ff_fact      = solver_state["K_ff_fact"]
    K_fc           = solver_state["K_fc"]

    # Read current Dirichlet values (which depend on time)
    _, dirichlet_vals = extract_dirichlet_bcs(mp, idx_ux, idx_uy, step_index)

    u = np.zeros(n_dof)
    if dirichlet_dofs.size > 0:
        u[dirichlet_dofs] = dirichlet_vals

    res_norm_free = 0.0

    if free_dofs.size > 0:
        # RHS for free DOFs
        rhs = - K_fc @ u[dirichlet_dofs]

        # Solve K_ff u_f = rhs using Cholesky factor
        u_f = cho_solve(K_ff_fact, rhs, check_finite=False)

        # Residual on free DOFs (optional check)
        # res = (K_ff_fact[0] @ u_f) - rhs  # not exactly K_ff, you’d need K_ff if you want exact residual
        # res_norm_free = float(np.linalg.norm(res))

        u[free_dofs] = u_f

    return u, res_norm_free

def homogenize_from_BC(mp, B_glob, gp_meta_B, C_by_props_id, u):
    """
    Given displacement vector u and B,C, compute:

      ε_B at GPs, σ_B at GPs, and homogenized strain/stress (3,).

    No Kratos GL/PK2 involved.
    """
    # 1) ε_B = B u
    eps_B = B_glob @ u            # (3*n_gp_total)
    eps_B_reshaped = eps_B.reshape(-1, 3)  # (n_gp_total, 3)

    # 2) σ_B = C ε_B
    s_from_B = np.zeros_like(eps_B_reshaped)
    for igp, (elem_id, igauss) in enumerate(gp_meta_B):
        elem = mp.Elements[elem_id]
        pid  = elem.Properties.Id
        C    = C_by_props_id[pid]
        s_from_B[igp, :] = C @ eps_B_reshaped[igp, :]

    # 3) Homogenization (same weighting as before)
    n_ips_dict = {}
    for elem_id, igauss in gp_meta_B:
        n_ips_dict[elem_id] = n_ips_dict.get(elem_id, 0) + 1

    area_dict = {}
    RVE_area = 0.0
    for elem in mp.Elements:
        area = elem.GetGeometry().Area()
        area_dict[elem.Id] = area
        RVE_area += area

    hom_strain_raw = np.zeros(3)
    hom_stress_raw = np.zeros(3)

    for igp, (elem_id, igauss) in enumerate(gp_meta_B):
        area = area_dict[elem_id]
        n_ips_elem = n_ips_dict[elem_id]
        w = area / n_ips_elem

        hom_strain_raw += w * eps_B_reshaped[igp, :]
        hom_stress_raw += w * s_from_B[igp, :]

    if RVE_area > 0.0:
        hom_strain = hom_strain_raw / RVE_area
        hom_stress = hom_stress_raw / RVE_area
    else:
        hom_strain = hom_strain_raw
        hom_stress = hom_stress_raw

    return hom_strain, hom_stress


# ==============================
# Custom AnalysisStage
# ==============================

class RVE_homogenization_dataset_generator(analysis_stage.AnalysisStage):
    """
    We still use an AnalysisStage only to:
      - load model (ModelPart, Properties, etc.)
      - run processes that apply BCs (fix DOFs and assign DISPLACEMENT_X/Y)
    The structural solver is NOT used.
    """

    def __init__(self, model, project_parameters):
        super().__init__(model, project_parameters)
        self.batch_strain = np.array([])

    def _CreateSolver(self):
        # We still create Kratos solver but never call SolveSolutionStep().
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
        """
        Here is where the macro-strain is turned into nodal displacements.
        We keep exactly Alejandro's logic, but Kratos won't solve anything,
        we just use the prescribed values as Dirichlet BCs.
        """
        super().ApplyBoundaryConditions()

        Ex  = self.batch_strain[0]
        Ey  = self.batch_strain[1]
        Exy = self.batch_strain[2]

        for node in self._GetSolver().GetComputingModelPart().Nodes:
            x_coord = node.X0
            y_coord = node.Y0
            displ_x = (Ex * x_coord + Exy * y_coord) * self.time / self.end_time
            displ_y = (Ey * y_coord + Exy * x_coord) * self.time / self.end_time
            displ_z = 0.0

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)


# ==============================
# Main: batches + our solver only
# ==============================

with open("ProjectParameters.json", 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

analysis_stage_module_name = parameters["analysis_stage"].GetString()
analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

analysis_stage_module = importlib.import_module(analysis_stage_module_name)
analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

log_lines = []

# Time stepping info (we will use the same as Kratos)
dt       = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
end_time = parameters["problem_data"]["end_time"].GetDouble()

theta = 0.0
phi   = 0.0
angle_increment   = 25.0
max_stretch_factor = 0.01  # lambda

all_strain_histories_BC = []
all_stress_histories_BC = []
batch = 0

while theta <= 360.0 + 1e-8:
    while phi <= 360.0 + 1e-8:
        batch += 1
        print(f"\n[INFO] Starting batch {batch} with theta={theta:.2f}, phi={phi:.2f}")

        global_model = KM.Model()
        simulation = RVE_homogenization_dataset_generator(global_model, parameters)

        # Same macro-strain as original script
        simulation.batch_strain = max_stretch_factor * np.array([
            np.cos(np.radians(phi)),                               # E_xx
            np.sin(np.radians(theta))  * np.cos(np.radians(phi)),  # E_yy
            (np.sin(np.radians(theta)) * np.sin(np.radians(phi))), # E_xy
        ])

        log_lines.append(
            f"Batch {batch}: theta={theta:.2f}, phi={phi:.2f}, strain={simulation.batch_strain.tolist()}"
        )

        # History for this batch
        strain_history_BC = [np.zeros(3)]
        stress_history_BC = [np.zeros(3)]

        simulation.Initialize()

        mp = simulation._GetSolver().GetComputingModelPart()
        idx_ux, idx_uy, _ = build_node_global_map(mp)
        B_glob, gp_meta_B = assemble_global_B(mp, idx_ux, idx_uy)
        C_by_props_id = build_C_matrices_from_structural_materials(STRUCTURAL_MATERIALS_FILE)
        K_glob = assemble_global_K_elemental(mp, idx_ux, idx_uy, C_by_props_id)
        # Precompute solver state (pattern + factorization)
        solver_state = prepare_dirichlet_solver_state(K_glob, mp, idx_ux, idx_uy, step_index=0)

        time = 0.0
        step = 0
        max_res_free = 0.0

        import time as tm

        # inside your batch time loop:
        while time < end_time - 1e-12:
            step += 1
            time += dt
            simulation.time = time
            simulation.step = step

            t_step_start = tm.time()

            # ------------------------------------------------------------
            # 1) Boundary condition application (Kratos processes)
            # ------------------------------------------------------------
            t_bc0 = tm.time()
            #simulation.InitializeSolutionStep()
            simulation.ApplyBoundaryConditions()
            t_bc1 = tm.time()
            bc_time = t_bc1 - t_bc0

            # ------------------------------------------------------------
            # 2) Our linear solver
            # ------------------------------------------------------------
            t_solve0 = tm.time()
            #u, res_free = solve_linear_system_with_dirichlet(K_glob, mp, idx_ux, idx_uy, step_index=0)
            u, res_free = solve_linear_system_with_dirichlet_fast(
                solver_state, mp, idx_ux, idx_uy, step_index=0
                )
            t_solve1 = tm.time()
            solve_time = t_solve1 - t_solve0

            max_res_free = max(max_res_free, res_free)

            #simulation.FinalizeSolutionStep()

            # ------------------------------------------------------------
            # 3) Homogenization from B,C only
            # ------------------------------------------------------------
            t_hom0 = tm.time()
            step_strain_BC, step_stress_BC = homogenize_from_BC(mp, B_glob, gp_meta_B, C_by_props_id, u)
            t_hom1 = tm.time()
            hom_time = t_hom1 - t_hom0

            strain_history_BC.append(step_strain_BC)
            stress_history_BC.append(step_stress_BC)

            t_step_end = tm.time()
            step_time = t_step_end - t_step_start

            # ------------------------------------------------------------
            # 4) Print timing summary for this step
            # ------------------------------------------------------------
            print(
                f"[Batch {batch} | Step {step:03d}] "
                f"BC={bc_time:.4f} s | "
                f"Solve={solve_time:.4f} s | "
                f"Homog={hom_time:.4f} s | "
                f"Total={step_time:.4f} s | "
                f"||res_free||={res_free:.3e}"
    )


        #simulation.Finalize()

        print(f"[BATCH {batch}] max ||K_ff u_f - rhs|| on free DOFs = {max_res_free:.3e}")

        all_strain_histories_BC.append(np.stack(strain_history_BC, axis=0))
        all_stress_histories_BC.append(np.stack(stress_history_BC, axis=0))

        phi += angle_increment

    phi = 0.0
    theta += angle_increment

# ==============================
# Save and simple plots
# ==============================

strain_tensor_BC = np.stack(all_strain_histories_BC, axis=0)
stress_tensor_BC = np.stack(all_stress_histories_BC, axis=0)

os.makedirs("data_set", exist_ok=True)
np.savez("data_set/all_stress_histories_BC_minimal.npz", stress=stress_tensor_BC)
np.savez("data_set/all_strain_histories_BC_minimal.npz", strain=strain_tensor_BC)

print("Results stored in:")
print("  data_set/all_stress_histories_BC_minimal.npz")
print("  data_set/all_strain_histories_BC_minimal.npz")

for batch_idx in range(stress_tensor_BC.shape[0]):
    Sxx = stress_tensor_BC[batch_idx, :, 0]
    Syy = stress_tensor_BC[batch_idx, :, 1]
    Sxy = stress_tensor_BC[batch_idx, :, 2]
    Exx = strain_tensor_BC[batch_idx, :, 0]
    Eyy = strain_tensor_BC[batch_idx, :, 1]
    Exy = strain_tensor_BC[batch_idx, :, 2]

    plt.figure()
    plt.plot(Exx, Sxx, marker='o', color='r', label="σ_xx (B,C)")
    plt.plot(Eyy, Syy, marker='o', color='b', label="σ_yy (B,C)")
    plt.plot(Exy, Sxy, marker='o', color='k', label="σ_xy (B,C)")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"Batch {batch_idx+1}: homogenized response (B,C only)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data_set/batch_{batch_idx+1}_stress_strain_BC_minimal.png")
    plt.close()

with open("data_set/batch_log_minimal.txt", "w") as f:
    f.write(f"Total batches: {batch}\n")
    f.write("Batch info (theta, phi, strain):\n")
    for line in log_lines:
        f.write(line + "\n")

print("Plots stored in data_set/batch_*_stress_strain_BC_minimal.png")
print("Log stored in data_set/batch_log_minimal.txt")

