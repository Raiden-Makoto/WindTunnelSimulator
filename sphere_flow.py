import os
os.environ["DDE_BACKEND"] = "torch"

import deepxde as dde #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import torch #type: ignore

rng = np.random.default_rng(6767)

# Physics Constants
rho = 1.0 # density of air
mu = rng.uniform(0.001, 0.01)
u_inlet = rng.uniform(1.0, 2.0)

# Domain Setup (2D Cross-Section of a Wind Tunnel)
tunnel = dde.geometry.Rectangle([0, 0], [2, 1])
sphere = dde.geometry.Disk([0.5, 0.5], 0.1) # Center=(0.5, 0.5), Radius=0.1 (i.e. sphere)
geom = dde.geometry.CSGDifference(tunnel, sphere)

# Navier-Stokes Equations for Incompressible Flow
def pde(x,y):
    # Inputs: x is coordinate (x,y)
    # Outputs: y is [u, v, p] (velocity_x, velocity_y, pressure)
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Automatic Differentiation (Calculating slopes instantly)
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    # The Equations (Momentum & Continuity)
    pde_u = (u * du_x + v * du_y) + (1/rho) * dp_x - mu * (du_xx + du_yy)
    pde_v = (u * dv_x + v * dv_y) + (1/rho) * dp_y - mu * (dv_xx + dv_yy)
    pde_cont = du_x + dv_y
    
    return [pde_u, pde_v, pde_cont]    

# Boundary Conditions
# Helper functions to identify where we are in the domain
def boundary_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def boundary_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], 2)
def boundary_walls(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))
def boundary_sphere(x, on_boundary):
    return on_boundary and not (
        np.isclose(x[0], 0) or np.isclose(x[0], 2) or
        np.isclose(x[1], 0) or np.isclose(x[1], 1)
    )