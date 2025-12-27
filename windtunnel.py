import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore

from potato import generate_random_potato
from pig import generate_flying_pig

# ===== CONFIGURATION =====
# Change this to switch shapes: "potato" or "pig"
SHAPE_TYPE = "pig"
# =========================

rng = np.random.default_rng(6767)

# Physics Constants
rho = 1.0 # density of air
mu = rng.uniform(0.001, 0.01)
u_inlet = rng.uniform(1.0, 2.0)

# Domain Setup (2D Cross-Section of a Wind Tunnel)
tunnel = dde.geometry.Rectangle([0, 0], [2, 1])

# Generate shape based on configuration
shape_generators = {
    "potato": generate_random_potato,
    "pig": generate_flying_pig
}

if SHAPE_TYPE not in shape_generators:
    raise ValueError(f"Unknown shape type: {SHAPE_TYPE}. Choose from: {list(shape_generators.keys())}")

print(f"Generating {SHAPE_TYPE}...")
shape_geom, shape_points = shape_generators[SHAPE_TYPE]()
geom = dde.geometry.CSGDifference(tunnel, shape_geom)

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
def boundary_shape(x, on_boundary):
    return on_boundary and not (
        np.isclose(x[0], 0) or np.isclose(x[0], 2) or
        np.isclose(x[1], 0) or np.isclose(x[1], 1)
    )

# Rule 1: Inlet Flow (Parabolic profile, fastest in middle)
bc_inlet_u = dde.icbc.DirichletBC(geom, lambda x: 4 * u_inlet * x[:, 1:2] * (1 - x[:, 1:2]), boundary_inlet, component=0)
bc_inlet_v = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_inlet, component=1)

# Rule 2: No-Slip Condition (Air sticks to walls and shape, velocity=0)
bc_walls_u = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_walls, component=0)
bc_walls_v = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_walls, component=1)
bc_shape_u = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_shape, component=0)
bc_shape_v = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_shape, component=1)

# Rule 3: Outlet (Pressure is zero, air leaves freely)
bc_outlet_p = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_outlet, component=2)

# Combine all Boundary Conditions
bcs = [bc_inlet_u, bc_inlet_v, bc_walls_u, bc_walls_v, bc_shape_u, bc_shape_v, bc_outlet_p]

# Train the PINN
data = dde.data.PDE(geom, pde, bcs, num_domain=2000, num_boundary=400, num_test=1000)
net = dde.nn.FNN([2] + [50] * 4 + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)

print("Compiling model...")
model.compile("adam", lr=1e-3)

print("Training...")
model.train(iterations=6767)

print("Generating Plot...")
x = np.linspace(0, 2, 200)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
xy = np.vstack((X.ravel(), Y.ravel())).T

# Ask the model: "What is the velocity at all these points?"
uvp = model.predict(xy)
u = uvp[:, 0].reshape(X.shape)
v = uvp[:, 1].reshape(X.shape)

plt.figure(figsize=(12, 6))
# Plot streamlines (flow lines) on top of velocity magnitude (color)
strm = plt.streamplot(X, Y, u, v, color=np.sqrt(u**2 + v**2), cmap="jet", density=1.5, linewidth=1)
plt.colorbar(strm.lines, label="Velocity Magnitude (m/s)")
plt.title(f"AI Prediction: Flow Past a {SHAPE_TYPE.capitalize()}")
plt.xlabel("Length (m)")
plt.ylabel("Height (m)")
plt.axis("equal")

# Draw the shape for reference
# Convert to numpy array and ensure it's closed
shape_points_array = np.array(shape_points)
shape_patch = plt.Polygon(shape_points_array, facecolor='white', edgecolor='black', linewidth=2, zorder=10)
plt.gca().add_patch(shape_patch)

plt.savefig(f"flow/{SHAPE_TYPE}_flow.png")