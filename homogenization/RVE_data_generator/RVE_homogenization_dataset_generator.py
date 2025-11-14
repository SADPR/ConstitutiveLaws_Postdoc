
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp
import numpy as np
import os
import matplotlib.pyplot as plt

"""
@file RVE_homogenization_dataset_generator.py
@brief Generator for RVE homogenization datasets using Kratos Multiphysics.

This script runs homogenization simulations on a Representative Volume Element (RVE)
for different combinations of applied strains (defined by theta and phi angles).
For each batch (combination of angles), it stores the homogenized strain and stress histories
in .npz files and generates plots of the stress-strain curves.
Additionally, it creates a log file with information about each batch.

- Saves stress and strain histories in: data_set/all_stress_histories.npz and data_set/all_strain_histories.npz
- Saves a log of the batches in: data_set/batch_log.txt
- Saves stress-strain plots for each batch in: data_set/batch_X_stress_strain_plots.png

@author Alejandro Cornejo
@date 2025

@section Usage
1. Configure the ProjectParameters.json file with the model parameters.
2. Run this script to generate the homogenization dataset.

@section Details
- Uses KratosMultiphysics and StructuralMechanicsApplication.
- Each batch corresponds to a combination of theta and phi.
- The history of each batch is stored as a tensor [batch, steps, voigt_size].
"""

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

        Ex = self.batch_strain[0]
        Ey = self.batch_strain[1]
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

    def CalculateHomogenizedStressAndStrain(self): # NOTE: it assumed that the BCs are already applied

        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        computing_model_part = self._GetSolver().GetComputingModelPart()

        for element in computing_model_part.Elements:
            dummy_strain = np.array(element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info))
            break # NOTE: this assumes that all elements have the same number of IP
        n_ips = dummy_strain.shape[0]
        voigt_size  = dummy_strain.shape[1]

        homogenized_stress = np.zeros(voigt_size)
        homogenized_strain  = np.zeros(voigt_size)
        RVE_area = 0.0

        for element in computing_model_part.Elements:
            strain = element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            stress = element.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)

            stress_vector_sum_ip = np.sum(np.array(stress), axis=0)
            strain_vector_sum_ip = np.sum(np.array(strain), axis=0)

            element_area = element.GetGeometry().Area()

            RVE_area += element_area
            homogenized_stress += element_area * stress_vector_sum_ip / n_ips
            homogenized_strain  += element_area * strain_vector_sum_ip / n_ips

        homogenized_stress /= RVE_area
        homogenized_strain /= RVE_area

        return homogenized_strain, homogenized_stress

#====================================================================================================
#====================================================================================================

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
max_stretch_factor  = 0.01 # lambda

# Here we will store the strain and stress histories for all batches
all_strain_histories = []
all_stress_histories = []

batch = 0

while theta <= 360.0:
    while phi <= 360.0:
        batch += 1
        # print(f"Batch {batch}:")
        # print(f"Theta: {theta}, Phi: {phi}")

        # NOTE: Each batch creates a new analysis_stage
        global_model = KM.Model()
        simulation = RVE_homogenization_dataset_generator(global_model, parameters)

        simulation.batch_strain = max_stretch_factor * np.array([
                np.cos(np.radians(phi)),                               # E_xx
                np.sin(np.radians(theta))  * np.cos(np.radians(phi)),  # E_yy
                (np.sin(np.radians(theta)) * np.sin(np.radians(phi))), # E_xy
        ])

        log_lines.append(
            f"Batch {batch}: theta={theta:.2f}, phi={phi:.2f}, strain={simulation.batch_strain.tolist()}"
        )

        # strains/stresses for one batch history, start with null values
        strain_history = [[0,0,0]]
        stress_history = [[0,0,0]]

        simulation.Initialize()
        while simulation.KeepAdvancingSolutionLoop():
            simulation.time = simulation._AdvanceTime()
            simulation.InitializeSolutionStep()
            simulation._GetSolver().Predict()
            is_converged = simulation._GetSolver().SolveSolutionStep()
            simulation.FinalizeSolutionStep()
            step_strain, step_stress = simulation.CalculateHomogenizedStressAndStrain()
            strain_history.append(step_strain)
            stress_history.append(step_stress)
        simulation.Finalize()

        all_strain_histories.append(np.stack(strain_history, axis=0))
        all_stress_histories.append(np.stack(stress_history, axis=0))

        phi += angle_increment

    phi = 0.0
    theta += angle_increment

strain_tensor = np.stack(all_strain_histories, axis=0)  # [batch_size, steps, voigt_size]
stress_tensor = np.stack(all_stress_histories, axis=0)  # [batch_size, steps, voigt_size]


# Printing and plotting the results
os.makedirs("data_set", exist_ok=True)
np.savez("data_set/all_stress_histories.npz", stress=stress_tensor)
np.savez("data_set/all_strain_histories.npz", strain=strain_tensor)
print("Results stored in data_set/all_stress_histories.npz and data_set/all_strain_histories.npz")

# plot for each batch
for batch_idx in range(stress_tensor.shape[0]):
    Sxx = stress_tensor[batch_idx, :, 0]
    Syy = stress_tensor[batch_idx, :, 1]
    Sxy = stress_tensor[batch_idx, :, 2]
    Exx = strain_tensor[batch_idx, :, 0]
    Eyy = strain_tensor[batch_idx, :, 1]
    Exy = strain_tensor[batch_idx, :, 2]

    plt.plot(Exx, Sxx, marker='o', color='r', label = "XX")
    plt.plot(Eyy, Syy, marker='o', color='b', label = "YY")
    plt.plot(Exy, Sxy, marker='o', color='k', label = "XY")
    plt.xlabel("Strain [-]")
    plt.ylabel("Stress [Pa]")
    plt.title(f"Batch {batch_idx+1}")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"data_set/batch_{batch_idx+1}_stress_strain_plots.png")
    plt.close()

with open("data_set/batch_log.txt", "w") as f:
    f.write(f"Total batches: {batch}\n")
    f.write("Batch info (theta, phi, strain):\n")
    for line in log_lines:
        f.write(line + "\n")