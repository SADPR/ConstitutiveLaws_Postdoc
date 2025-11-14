#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import KratosMultiphysics as KM
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers

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

def set_affine_nodal_displacements(model_part, A):
    """
    Impose u_I = A * [X_I, Y_I]^T on the CURRENT solution step (no solve needed).
    A: 2x2 numpy array.
    """
    for node in model_part.Nodes:
        X, Y = node.X0, node.Y0
        ux = A[0,0]*X + A[0,1]*Y
        uy = A[1,0]*X + A[1,1]*Y
        node.SetSolutionStepValue(KM.DISPLACEMENT_X, ux)
        node.SetSolutionStepValue(KM.DISPLACEMENT_Y, uy)

def test_B_affine_field(model_part, A, tol=1e-10, print_mismatches=10):
    """
    Verifies that B*d reproduces sym(A) everywhere for an affine field u(X)=A X.
    Compares Voigt(3) engineering strain: [εxx, εyy, γxy] vs [A11, A22, A12+A21].
    Returns (max_error, list_of_mismatches).
    """
    # Set nodal displacements to the affine field
    set_affine_nodal_displacements(model_part, A)

    target = np.array([A[0,0], A[1,1], A[0,1] + A[1,0]])  # [εxx, εyy, γxy] expected

    max_err = 0.0
    n_checked = 0
    bad = []  # (elem_id, igauss, eps_B, target, abs_err)

    for elem in model_part.Elements:
        geom = elem.GetGeometry()

        # Element DOF vector d^e = [ux1, uy1, ux2, uy2, ...]
        de = []
        for node in geom:
            de.append(node.GetSolutionStepValue(KM.DISPLACEMENT_X))
            de.append(node.GetSolutionStepValue(KM.DISPLACEMENT_Y))
        de = np.array(de)

        # Number of integration points = number of rows in N table
        Ns = np.array(geom.ShapeFunctionsValues())   # shape: (n_ip, nnode)
        n_ip = Ns.shape[0]

        for igauss in range(n_ip):
            # Local derivatives and Jacobian
            DNDe = np.array(geom.ShapeFunctionDerivatives(1, igauss))  # (nnode,2) [ξ,η]
            J = np.array(geom.Jacobian(igauss))                        # (2,2)
            invJ = np.linalg.inv(J)
            DNDX = DNDe @ invJ                                         # (nnode,2) [x,y]

            # B and strain
            B = build_B_from_DNDX(DNDX)                                # (3,2*nnode)
            eps_B = B @ de                                             # (3,)
            print(eps_B)
            print(target)
            print("-------")
            err = np.abs(eps_B - target)
            emax = err.max()
            max_err = max(max_err, emax)
            n_checked += 1
            if emax > tol:
                bad.append((elem.Id, igauss, eps_B.copy(), target.copy(), err.copy()))

    print(f"[B-test] Checked {n_checked} Gauss points | max |error| = {max_err:.3e}")
    if bad:
        print(f"  First {min(print_mismatches,len(bad))} mismatches:")
        for (eid, ig, eps_B, tgt, err) in bad[:print_mismatches]:
            print(f"   - elem {eid}, gp {ig}: eps_B={eps_B}, target={tgt}, |Δ|={err}")

    return max_err, bad

# ==============================
# Main: load model, run test
# ==============================

def main():
    # 1) Read ProjectParameters.json
    with open("ProjectParameters.json", "r") as f:
        parameters = KM.Parameters(f.read())

    # 2) Build Model and Solver (no full analysis stage needed)
    model = KM.Model()
    solver = structural_solvers.CreateSolver(model, parameters)

    # Standard solver bootstrap to have DOFs & model part populated
    solver.AddVariables()
    solver.ImportModelPart()
    solver.PrepareModelPart()
    solver.AddDofs()
    # No need to Initialize() or solve — we are only testing B with an imposed u(X).

    # 3) Get the computing model part
    model_part = solver.GetComputingModelPart()
    print("[Info] Model part name:", model_part.Name)
    print("[Info] #Nodes:", model_part.NumberOfNodes(), "| #Elements:", model_part.NumberOfElements())

    # 4) Choose an affine field A (small magnitudes recommended)
    # Examples:
    alpha = 1e-3
    beta  = 2e-3
    s     = 1e-3
    theta = 1e-3

    # a) Uniaxial X → expect [alpha, 0, 0]
    # A = np.array([[alpha, 0.0],
    #               [0.0,   0.0]])

    # b) Uniaxial Y → expect [0, beta, 0]
    # A = np.array([[0.0,   0.0],
    #               [0.0,   beta]])

    # c) Symmetric shear → expect [0, 0, 2s]
    # A = np.array([[0.0, s],
    #               [s,   0.0]])

    # d) Pure rotation (rigid) → expect [0, 0, 0]
    # A = np.array([[0.0,    -theta],
    #               [theta,   0.0]])

    # Pick one:
    A = np.array([[alpha, s],
                   [s,   beta]]) 

    print("[Info] Testing with affine A =\n", A)

    # 5) Run the B-test
    tol = 1e-10
    max_err, bad_spots = test_B_affine_field(model_part, A, tol=tol, print_mismatches=10)

    # 6) Simple pass/fail message
    if max_err <= tol:
        print(f"[PASS] B reproduces the affine field within tol={tol:g}.")
    else:
        print(f"[WARN] Max error {max_err:.3e} exceeds tol={tol:g}. See mismatches above.")

if __name__ == "__main__":
    main()
