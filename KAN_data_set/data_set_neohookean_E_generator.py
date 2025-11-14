import numpy as np
import matplotlib.pyplot as pl
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ConstitutiveLawsApplication    as CLA
import json
import math
# import scienceplots

#====================================================================================
#====================================================================================

def CreateGeometry(model_part, dim):
    # Create new nodes
    node1 = model_part.CreateNewNode(1, 0.0, 0.0, 0.0)
    node2 = model_part.CreateNewNode(2, 1.0, 0.0, 0.0)
    node3 = model_part.CreateNewNode(3, 0.0, 1.0, 0.0)

    if (dim == 2):
        nnodes = 3

        # Allocate a geometry
        geom = KM.Triangle2D3(node1,node2,node3)
    elif (dim == 3):
        nnodes = 4
        node4 = model_part.CreateNewNode(4, 0.0, 0.0, 1.0)

        # Allocate a geometry
        geom = KM.Tetrahedra3D4(node1,node2,node3,node4)
    else:
        raise Exception("Error: bad dimension value: ", dim)
    return [geom, nnodes]

#====================================================================================
#====================================================================================

def _set_cl_parameters(cl_options, F, detF, strain_vector, stress_vector, constitutive_matrix, N, DN_DX, model_part, properties, geom):
    # Setting the parameters - note that a constitutive law may not need them all!
    cl_params = KM.ConstitutiveLawParameters()
    cl_params.SetOptions(cl_options)
    cl_params.SetDeformationGradientF(F)
    cl_params.SetDeterminantF(detF)
    cl_params.SetStrainVector(strain_vector)
    cl_params.SetStressVector(stress_vector)
    cl_params.SetConstitutiveMatrix(constitutive_matrix)
    cl_params.SetShapeFunctionsValues(N)
    cl_params.SetShapeFunctionsDerivatives(DN_DX)
    cl_params.SetProcessInfo(model_part.ProcessInfo)
    cl_params.SetMaterialProperties(properties)
    cl_params.SetElementGeometry(geom)

    ## Do all sort of checks
    cl_params.CheckAllParameters() # Can not use this until the geometry is correctly exported to python
    cl_params.CheckMechanicalVariables()
    cl_params.CheckShapeFunctions()
    return cl_params

#====================================================================================
#====================================================================================

def CalculateStressFromE(ConstitutiveLaw, Properties, Geometry, ModelPart, E):
    # The selected CL
    cl = ConstitutiveLaw

    dimension  = cl.WorkingSpaceDimension()
    voigt_size = cl.GetStrainSize()

    N = KM.Vector()
    DN_DX = KM.Matrix()

    cl.Check(Properties, Geometry, ModelPart.ProcessInfo)

    cl_options = KM.Flags()
    cl_options.Set(KM.ConstitutiveLaw.USE_ELEMENT_PROVIDED_STRAIN, True)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_CONSTITUTIVE_TENSOR, False)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_STRESS, True)

    # Define deformation gradient
    F = KM.Matrix(dimension, dimension)
    detF = 1.0

    strain_vector = E
    stress_vector = KM.Vector(voigt_size)
    constitutive_matrix = KM.Matrix(voigt_size,voigt_size)

    # Let's initialize the values
    stress_vector.fill(0.0)
    F.fill(0.0)
    constitutive_matrix.fill(0.0)
    # s = np.array(stress_vector, copy=False)

    # Setting the parameters - note that a constitutive law may not need them all!
    cl_params = _set_cl_parameters(cl_options, F, detF, strain_vector, stress_vector, constitutive_matrix, N, DN_DX, ModelPart, Properties, Geometry)

    # We run them in case we need to account for history
    cl.InitializeMaterialResponsePK2(cl_params)
    cl.CalculateMaterialResponsePK2(cl_params)
    cl.FinalizeMaterialResponsePK2(cl_params)

    # return strain_vector, stress_vector
    return cl_params.GetStressVector()

#====================================================================================
#====================================================================================

def ResetE(E):
    E[0] = 0.0
    E[1] = 0.0
    E[2] = 0.0

#====================================================================================
#====================================================================================

case_number = 1
current_model = KM.Model()
model_part = current_model.CreateModelPart("NeoHookean")

cl = CLA.HyperElasticPlaneStrain2DLaw()
dimension = cl.WorkingSpaceDimension()
voigt_size = cl.GetStrainSize()

properties = KM.Properties(1)
Young = 1e7
nu = 0.4
properties.SetValue(KM.YOUNG_MODULUS, Young)
properties.SetValue(KM.POISSON_RATIO, nu)
properties.SetValue(KM.CONSTITUTIVE_LAW, cl)

[geometry, nnodes] = CreateGeometry(model_part, dimension)

# Initialize the Green-Lagrange strain vector
E = KM.Vector(voigt_size)

# Initialize the material
cl.InitializeMaterial(properties, geometry, KM.Vector(nnodes))

# ----- Start the case loop
angle_increment_deg = 20.0
n_steps             = 25 # step in the loading history
max_stretch_factor  = 0.25 # lambda
theta               = 0.0
phi                 = 0.0
# -----
strains_matrix = np.array((0.0, 0.0, 0.0))

with open("neo_hookean_hyperelastic_law/data_set.log", "w") as file:
    file.write("Dataset generated for a HyperElasticPlaneStrain2DLaw constitutive law \n\n")
    file.write("Data used for the generation: \n")
    file.write("\tYoung's Modulus = " + str(Young) + "\n")
    file.write("\tPoisson's ratio = " + str(nu) + "\n")
    file.write("\tAngle increment = " + str(angle_increment_deg) + " deg \n")
    file.write("\tNumber of steps per load history = " + str(n_steps) + "\n\n")

strain_history = np.zeros((n_steps, voigt_size))
stress_history = strain_history.copy()

while theta <= 360.0:
    while phi <= 360.0:
        dExx = math.cos(math.radians(theta)) * max_stretch_factor
        dEyy = math.sin(math.radians(theta)) * math.cos(math.radians(phi)) * max_stretch_factor
        dExy = 2.0 * math.sin(math.radians(theta)) * math.sin(math.radians(phi)) * max_stretch_factor

        strains_matrix = np.vstack((strains_matrix, np.array((dExx, dEyy, dExy))))

        for step in range(0, n_steps):
            alpha = step / n_steps
            E[0] = dExx * alpha
            E[1] = dEyy * alpha
            E[2] = dExy * alpha

            stress = CalculateStressFromE(cl, properties, geometry, model_part, E)
            strain_history[step, :] = E
            stress_history[step, :] = stress

        output_type = "no_plot"
        if output_type == "plot":
            #pl.style.use('science')
            name = "neo_hookean_hyperelastic_law/strain_stress_plots/strain_stress_data_case_" + str(case_number) + ".png"
            title = "theta = " + str(theta) + " ; phi = " + str(phi)
            # title = r"$\theta$ = " + str(theta) + r" ; $\phi$ = " + str(phi)
            title = "theta = " + str(theta) + " ; phi = " + str(phi)
            pl.plot(strain_history[:, 0], stress_history[:, 0], label="Ground truth \varepsilon_{xx}", marker='X', color="k",  markersize=2, markerfacecolor='none')
            pl.plot(strain_history[:, 1], stress_history[:, 1], label="Ground truth \varepsilon_{yy}", marker='X', color="r",  markersize=2, markerfacecolor='none')
            pl.plot(strain_history[:, 2], stress_history[:, 2], label="Ground truth \gamma_{xy}",      marker='X', color="b",  markersize=2, markerfacecolor='none')
            pl.xlabel("Green-Lagrange Strain [-]")
            pl.ylabel("PK2 Stress [Pa]")
            pl.title(title)
            pl.legend(loc='best')
            pl.grid()
            pl.savefig(name, dpi=300, bbox_inches=None)
            pl.close()

            name = "neo_hookean_hyperelastic_law/strain_histories_plots/strain_history_case_" + str(case_number) + ".png"
            fig = pl.figure()
            #pl.style.use('default')
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(strain_history[:, 0], strain_history[:, 1], strain_history[:, 2], c='r', marker='o', label="Strain history")
            ax.set_xlabel("\varepsilon_{xx}")
            ax.set_ylabel("\varepsilon_{yy}")
            ax.set_zlabel("\gamma_{xy}")

            pl.title(title)
            pl.savefig(name, dpi=300, bbox_inches=None)
            pl.close()

        with open("neo_hookean_hyperelastic_law/data_set.log", "a") as file:
            strain = strain_history[n_steps-1, :]
            file.write("CASE " + str(case_number) + "\n")
            file.write("\tImposed E = " + str(strain) + "\n")
            file.write("\tTheta = " + str(theta) + ", Phi = " + str(phi) + "\n")
            file.write("\tNorm of E = " + str(math.sqrt(strain[0]**2 + strain[1]**2 + (0.5 * strain[2])**2)) + "\n\n")

        name = "neo_hookean_hyperelastic_law/raw_data/E_S_data_case_" + str(case_number) + ".npz"
        np.savez(name, strain_history = strain_history, stress_history = stress_history)
        print("\t --> Case: ", case_number, "Data saved to ", name)

        '''
        NOTE:
        Then we can load them by

        loaded_data = np.load(name)

        loaded_strain_history = loaded_data["strain_history"]
        loaded_stress_history = loaded_data["stress_history"]
        '''

        case_number += 1
        phi += angle_increment_deg
        if theta == 0.0 or theta == 360.0:
            break
    theta += angle_increment_deg
    phi = 0.0

# Print the sphere os strains
name = "neo_hookean_hyperelastic_law/strain_sphere.png"
fig = pl.figure()
pl.style.use('default')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(strains_matrix[:, 0], strains_matrix[:, 1], strains_matrix[:, 2], c='green', marker='o', label="Strain Cases")
ax.set_xlabel(r"$\varepsilon_{xx}$")
ax.set_ylabel(r"$\varepsilon_{yy}$")
ax.set_zlabel(r"$\gamma_{xy}$")
pl.savefig(name, dpi=300, bbox_inches=None)
pl.close()

#====================================================================================
#====================================================================================
