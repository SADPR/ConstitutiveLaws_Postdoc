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

"""
RVE homogenization + B,C consistency check.

- Kratos solves the RVE problem as in Alejandro's script.
- At each time step, we:
  * Build ε_B = B u at all Gauss points (global operator B).
  * Build σ_B = C ε_B, with C taken from StructuralMaterials.json per properties_id.
  * Compare ε_B vs GREEN_LAGRANGE_STRAIN_VECTOR and σ_B vs PK2_STRESS_VECTOR.
  * Compute homogenized strain/stress from (ε_B, σ_B) in the same way as the
    original CalculateHomogenizedStressAndStrain.

We store:
- Kratos homogenized histories: strain_tensor_kratos, stress_tensor_kratos
- B,C homogenized histories:   strain_tensor_BC,     stress_tensor_BC
and generate comparison plots for each batch.
"""

# ==============================
# Global config / files
# ==============================
REL_EPS = 1e-14
STRUCTURAL_MATERIALS_FILE = "StructuralMaterials.json"

# ==============================
# Small helpers
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


def compute_BC_homogenized_and_errors(mp, B_glob, gp_meta_B, C_by_props_id):
    """
    For the CURRENT displacement field in 'mp':
      - Build ε_B = B u  at all GPs.
      - Build σ_B = C ε_B per material (via properties_id).
      - Read Kratos GREEN_LAGRANGE_STRAIN_VECTOR and PK2_STRESS_VECTOR at GPs.
      - Compute:
          * max |GL - ε_B|
          * max |PK2 - σ_B|
          * max relative |PK2 - σ_B|
      - Compute homogenized strain/stress from (ε_B, σ_B) in the same way as
        Alejandro's CalculateHomogenizedStressAndStrain.

    Returns:
        hom_strain_BC  : (3,)
        hom_stress_BC  : (3,)
        max_err_GL_vs_B: float
        max_err_PK2_vs_B: float
        max_rel_PK2_vs_B: float
    """
    process_info = mp.ProcessInfo

    # 1) Build global displacement vector u
    ta_disp = KM.TensorAdaptors.HistoricalVariableTensorAdaptor(
        mp.Nodes, KM.DISPLACEMENT, data_shape=[2]
    )
    ta_disp.CollectData()
    u = ta_disp.data.reshape(-1)  # [ux1, uy1, ux2, uy2, ...]

    # 2) ε_B = B u
    eps_B = B_glob.dot(u) if hasattr(B_glob, "dot") else (B_glob @ u)
    eps_B_reshaped = eps_B.reshape(-1, 3)  # (n_gp_total, 3)

    n_gp_total = eps_B_reshaped.shape[0]

    # 3) σ_B = C ε_B (per material)
    s_from_B = np.zeros_like(eps_B_reshaped)
    for igp, (elem_id, igauss) in enumerate(gp_meta_B):
        elem = mp.Elements[elem_id]
        pid  = elem.Properties.Id

        C = C_by_props_id[pid]
        s_from_B[igp, :] = C @ eps_B_reshaped[igp, :]

    # 4) Kratos GL and PK2 at GPs via TensorAdaptors
    gp_gl = KM.TensorAdaptors.GaussPointVariableTensorAdaptor(
        mp.Elements, KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info
    )
    gp_gl.Check()
    gp_gl.CollectData()
    gl_data  = gp_gl.data
    gl_shape = gp_gl.DataShape()
    voigt_size_gl = gl_shape[1]
    gl_reshaped = gl_data.reshape(-1, voigt_size_gl)[:, :3]  # (n_gp_total, 3)

    gp_pk2 = KM.TensorAdaptors.GaussPointVariableTensorAdaptor(
        mp.Elements, KM.PK2_STRESS_VECTOR, process_info
    )
    gp_pk2.Check()
    gp_pk2.CollectData()
    pk2_data  = gp_pk2.data
    pk2_shape = gp_pk2.DataShape()
    voigt_size_pk2 = pk2_shape[1]
    pk2_reshaped = pk2_data.reshape(-1, voigt_size_pk2)[:, :3]

    # 5) Errors at Gauss points
    err_GL_vs_B   = np.abs(gl_reshaped - eps_B_reshaped)
    max_err_GL_vs_B = float(err_GL_vs_B.max()) if n_gp_total > 0 else 0.0

    err_PK2_vs_B  = np.abs(pk2_reshaped - s_from_B)
    max_err_PK2_vs_B = float(err_PK2_vs_B.max()) if n_gp_total > 0 else 0.0

    rel_PK2_vs_B  = rel_err(pk2_reshaped, s_from_B)
    max_rel_PK2_vs_B = float(rel_PK2_vs_B.max()) if n_gp_total > 0 else 0.0

    # 6) Homogenized strain/stress from (ε_B, σ_B)
    #    Equivalent to Alejandro's method:
    #    - each element: area * (sum over IP / n_ips)
    #    - divide total by RVE_area

    # First, count n_ips per element from gp_meta_B
    n_ips_dict = {}
    for elem_id, igauss in gp_meta_B:
        n_ips_dict[elem_id] = n_ips_dict.get(elem_id, 0) + 1

    # Element areas
    area_dict = {}
    RVE_area = 0.0
    for elem in mp.Elements:
        area = elem.GetGeometry().Area()
        area_dict[elem.Id] = area
        RVE_area += area

    hom_strain_raw = np.zeros(3)
    hom_stress_raw = np.zeros(3)

    # Accumulate area * (sum_ip / n_ips) by distributing weights to each GP
    for igp, (elem_id, igauss) in enumerate(gp_meta_B):
        area = area_dict[elem_id]
        n_ips_elem = n_ips_dict[elem_id]
        w = area / n_ips_elem  # this matches "area * (sum_ip / n_ips)" once summed over IPs

        hom_strain_raw += w * eps_B_reshaped[igp, :]
        hom_stress_raw += w * s_from_B[igp, :]

    if RVE_area > 0.0:
        hom_strain_BC = hom_strain_raw / RVE_area
        hom_stress_BC = hom_stress_raw / RVE_area
    else:
        hom_strain_BC = hom_strain_raw
        hom_stress_BC = hom_stress_raw

    return hom_strain_BC, hom_stress_BC, max_err_GL_vs_B, max_err_PK2_vs_B, max_rel_PK2_vs_B


# ==============================
# Custom AnalysisStage
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
        self._list_of_processes.extend(self._list_of_output_processes) # Adding the output processes to the regular processes
        self._list_of_output_processes.extend(deprecated_output_processes)

    def ApplyBoundaryConditions(self):
        super().ApplyBoundaryConditions()

        Ex  = self.batch_strain[0]
        Ey  = self.batch_strain[1]
        Exy = self.batch_strain[2]

        for node in self._GetSolver().GetComputingModelPart().Nodes:
            # NOTE: here we assume that one corner of the RVE is at (0,0,0)
            x_coord = node.X0
            y_coord = node.Y0
            z_coord = node.Z0
            displ_x = (Ex * x_coord + Exy * y_coord) * self.time / self.end_time
            displ_y = (Ey * y_coord + Exy * x_coord) * self.time / self.end_time
            displ_z = 0.0  # Assuming no displacement in Z direction for 2D RVE

            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)

    def CalculateHomogenizedStressAndStrain(self):
        """
        Original Alejandro's homogenization using element.CalculateOnIntegrationPoints.
        Used here as the "Kratos reference" for comparison.
        """
        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        computing_model_part = self._GetSolver().GetComputingModelPart()

        for element in computing_model_part.Elements:
            dummy_strain = np.array(element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info))
            break  # NOTE: assumes all elements have the same number of IP
        n_ips = dummy_strain.shape[0]
        voigt_size  = dummy_strain.shape[1]

        homogenized_stress = np.zeros(voigt_size)
        homogenized_strain = np.zeros(voigt_size)
        RVE_area = 0.0

        for element in computing_model_part.Elements:
            strain = element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            stress = element.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)

            stress_vector_sum_ip = np.sum(np.array(stress), axis=0)
            strain_vector_sum_ip = np.sum(np.array(strain), axis=0)

            element_area = element.GetGeometry().Area()

            RVE_area += element_area
            homogenized_stress += element_area * stress_vector_sum_ip / n_ips
            homogenized_strain += element_area * strain_vector_sum_ip / n_ips

        homogenized_stress /= RVE_area
        homogenized_strain /= RVE_area

        return homogenized_strain, homogenized_stress


# ==============================
# Main: batch loops + B,C checks
# ==============================

with open("ProjectParameters.json", 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

analysis_stage_module_name = parameters["analysis_stage"].GetString()
analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

analysis_stage_module = importlib.import_module(analysis_stage_module_name)
analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

log_lines = []

theta = 0.0
phi = 0.0
angle_increment = 25.0
max_stretch_factor  = 0.01  # lambda

# Here we will store the strain and stress histories for all batches
all_strain_histories_kratos = []
all_stress_histories_kratos = []

all_strain_histories_BC = []  # from B,C
all_stress_histories_BC = []

batch = 0

while theta <= 360.0 + 1e-8:
    while phi <= 360.0 + 1e-8:
        batch += 1
        print(f"\n[INFO] Starting batch {batch} with theta={theta:.2f}, phi={phi:.2f}")

        # NOTE: Each batch creates a new analysis_stage
        global_model = KM.Model()
        simulation = RVE_homogenization_dataset_generator(global_model, parameters)

        # Define macro-strain vector for this batch
        simulation.batch_strain = max_stretch_factor * np.array([
            np.cos(np.radians(phi)),                               # E_xx
            np.sin(np.radians(theta))  * np.cos(np.radians(phi)),  # E_yy
            (np.sin(np.radians(theta)) * np.sin(np.radians(phi))), # E_xy
        ])

        log_lines.append(
            f"Batch {batch}: theta={theta:.2f}, phi={phi:.2f}, strain={simulation.batch_strain.tolist()}"
        )

        # Histories: Kratos and B,C
        strain_history_kratos = [np.zeros(3)]
        stress_history_kratos = [np.zeros(3)]

        strain_history_BC = [np.zeros(3)]
        stress_history_BC = [np.zeros(3)]

        simulation.Initialize()

        # Build B_glob, gp_meta_B, and C_by_props_id ONCE for this batch
        mp = simulation._GetSolver().GetComputingModelPart()
        idx_ux, idx_uy, _ = build_node_global_map(mp)
        B_glob, gp_meta_B = assemble_global_B(mp, idx_ux, idx_uy)
        C_by_props_id = build_C_matrices_from_structural_materials(STRUCTURAL_MATERIALS_FILE)

        # For diagnostics: max errors over all steps in this batch
        batch_max_GL_vs_B   = 0.0
        batch_max_PK2_vs_B  = 0.0
        batch_max_rel_PK2_B = 0.0

        while simulation.KeepAdvancingSolutionLoop():
            simulation.time = simulation._AdvanceTime()
            simulation.InitializeSolutionStep()
            simulation._GetSolver().Predict()
            is_converged = simulation._GetSolver().SolveSolutionStep()
            simulation.FinalizeSolutionStep()

            # 1) Kratos homogenized (reference)
            step_strain_K, step_stress_K = simulation.CalculateHomogenizedStressAndStrain()
            strain_history_kratos.append(step_strain_K)
            stress_history_kratos.append(step_stress_K)

            # 2) B,C-based homogenized and GP-level consistency errors
            step_strain_BC, step_stress_BC, \
                max_GL, max_PK2, max_rel_PK2 = compute_BC_homogenized_and_errors(
                    mp, B_glob, gp_meta_B, C_by_props_id
            )

            strain_history_BC.append(step_strain_BC)
            stress_history_BC.append(step_stress_BC)

            batch_max_GL_vs_B   = max(batch_max_GL_vs_B,   max_GL)
            batch_max_PK2_vs_B  = max(batch_max_PK2_vs_B,  max_PK2)
            batch_max_rel_PK2_B = max(batch_max_rel_PK2_B, max_rel_PK2)

        simulation.Finalize()

        print(f"[BATCH {batch}] max |GL - B u|       = {batch_max_GL_vs_B:.3e}")
        print(f"[BATCH {batch}] max |PK2 - C(B u)|   = {batch_max_PK2_vs_B:.3e}")
        print(f"[BATCH {batch}] max rel|PK2 - C(B u)|= {batch_max_rel_PK2_B:.3e}")

        all_strain_histories_kratos.append(np.stack(strain_history_kratos, axis=0))
        all_stress_histories_kratos.append(np.stack(stress_history_kratos, axis=0))

        all_strain_histories_BC.append(np.stack(strain_history_BC, axis=0))
        all_stress_histories_BC.append(np.stack(stress_history_BC, axis=0))

        phi += angle_increment

    phi = 0.0
    theta += angle_increment

# ==============================
# Pack tensors and save
# ==============================

strain_tensor_kratos = np.stack(all_strain_histories_kratos, axis=0)
stress_tensor_kratos = np.stack(all_stress_histories_kratos, axis=0)

strain_tensor_BC = np.stack(all_strain_histories_BC, axis=0)
stress_tensor_BC = np.stack(all_stress_histories_BC, axis=0)

os.makedirs("data_set", exist_ok=True)

np.savez("data_set/all_stress_histories_kratos.npz", stress=stress_tensor_kratos)
np.savez("data_set/all_strain_histories_kratos.npz", strain=strain_tensor_kratos)

np.savez("data_set/all_stress_histories_BC.npz", stress=stress_tensor_BC)
np.savez("data_set/all_strain_histories_BC.npz", strain=strain_tensor_BC)

print("Results stored in:")
print("  data_set/all_stress_histories_kratos.npz")
print("  data_set/all_strain_histories_kratos.npz")
print("  data_set/all_stress_histories_BC.npz")
print("  data_set/all_strain_histories_BC.npz")

# ==============================
# Plots per batch: Kratos vs B,C
# ==============================

for batch_idx in range(stress_tensor_kratos.shape[0]):
    Sxx_K = stress_tensor_kratos[batch_idx, :, 0]
    Syy_K = stress_tensor_kratos[batch_idx, :, 1]
    Sxy_K = stress_tensor_kratos[batch_idx, :, 2]
    Exx_K = strain_tensor_kratos[batch_idx, :, 0]
    Eyy_K = strain_tensor_kratos[batch_idx, :, 1]
    Exy_K = strain_tensor_kratos[batch_idx, :, 2]

    Sxx_B = stress_tensor_BC[batch_idx, :, 0]
    Syy_B = stress_tensor_BC[batch_idx, :, 1]
    Sxy_B = stress_tensor_BC[batch_idx, :, 2]
    Exx_B = strain_tensor_BC[batch_idx, :, 0]
    Eyy_B = strain_tensor_BC[batch_idx, :, 1]
    Exy_B = strain_tensor_BC[batch_idx, :, 2]

    plt.figure()
    plt.plot(Exx_K, Sxx_K, marker='o', linestyle='-',  color='r', label="σ_xx Kratos")
    plt.plot(Exx_B, Sxx_B, marker='x', linestyle='--', color='r', label="σ_xx B,C")

    plt.plot(Eyy_K, Syy_K, marker='o', linestyle='-',  color='b', label="σ_yy Kratos")
    plt.plot(Eyy_B, Syy_B, marker='x', linestyle='--', color='b', label="σ_yy B,C")

    plt.plot(Exy_K, Sxy_K, marker='o', linestyle='-',  color='k', label="σ_xy Kratos")
    plt.plot(Exy_B, Sxy_B, marker='x', linestyle='--', color='k', label="σ_xy B,C")

    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"Batch {batch_idx+1}: Kratos vs B,C homogenized response")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data_set/batch_{batch_idx+1}_stress_strain_compare_BC.png")
    plt.close()

# Log file
with open("data_set/batch_log.txt", "w") as f:
    f.write(f"Total batches: {batch}\n")
    f.write("Batch info (theta, phi, strain):\n")
    for line in log_lines:
        f.write(line + "\n")

print("Plots stored in data_set/batch_*_stress_strain_compare_BC.png")
print("Log stored in data_set/batch_log.txt")
