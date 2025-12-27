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
rho, mu, u_inlet = 1.0, 0.002, 1.0  # <--- Lower viscosity
thermal_diff = 0.02  # Thermal diffusivity (how fast heat spreads)

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

# Navier-Stokes Equations + Heat Equation
def pde(x, y):
    # Inputs: x is coordinate (x,y)
    # Outputs: y is [u, v, p, T] (velocity_x, velocity_y, pressure, temperature)
    u, v, p, T = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    
    # Automatic Differentiation (Calculating slopes instantly)
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    dT_x = dde.grad.jacobian(y, x, i=3, j=0)
    dT_y = dde.grad.jacobian(y, x, i=3, j=1)
    
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    dT_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
    dT_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)
    
    # The Equations (Momentum & Continuity)
    pde_u = (u * du_x + v * du_y) + (1/rho) * dp_x - mu * (du_xx + du_yy)
    pde_v = (u * dv_x + v * dv_y) + (1/rho) * dp_y - mu * (dv_xx + dv_yy)
    pde_cont = du_x + dv_y
    
    # Heat Equation: (u*dT/dx + v*dT/dy) = D * (d2T/dx2 + d2T/dy2)
    # "Wind moving heat" = "Heat spreading naturally"
    pde_temp = (u * dT_x + v * dT_y) - thermal_diff * (dT_xx + dT_yy)
    
    return [pde_u, pde_v, pde_cont, pde_temp]    

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

# Thermal Boundary Conditions
# Rule 4: Shape is HOT (Temperature = 1.0)
bc_shape_T = dde.icbc.DirichletBC(geom, lambda x: 1.0, boundary_shape, component=3)
# Rule 5: Incoming wind is COLD (Temperature = 0.0)
bc_inlet_T = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_inlet, component=3)
# Rule 6: Walls are insulated (Temperature = 0.0)
bc_walls_T = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_walls, component=3)

# Combine all Boundary Conditions
bcs = [bc_inlet_u, bc_inlet_v, bc_walls_u, bc_walls_v, bc_shape_u, bc_shape_v, bc_outlet_p,
       bc_shape_T, bc_inlet_T, bc_walls_T]

# Train the PINN
# We need more points to capture the sharper turbulence
data = dde.data.PDE(geom, pde, bcs, num_domain=5000, num_boundary=1000, num_test=1000)

# We need a larger network for harder physics (now with 4 outputs: u, v, p, T)
net = dde.nn.FNN([2] + [64] * 5 + [4], "tanh", "Glorot normal")  # Slightly deeper network
model = dde.Model(data, net)

print("Compiling model...")
model.compile("adam", lr=1e-3)

print("Training...")
# It takes longer to learn chaos
model.train(iterations=12000)

print("Generating Plot...")
x = np.linspace(0, 2, 300)
y = np.linspace(0, 1, 150)
X, Y = np.meshgrid(x, y)
xy = np.vstack((X.ravel(), Y.ravel())).T

# Ask the model: "What is the velocity, pressure, and temperature at all these points?"
output = model.predict(xy)
u = output[:, 0].reshape(X.shape)
v = output[:, 1].reshape(X.shape)
temp = output[:, 3].reshape(X.shape)  # Index 3 is Temperature

plt.figure(figsize=(12, 6))
# Plot Temperature as color map (background)
plt.contourf(X, Y, temp, levels=100, cmap="inferno", zorder=1)
cbar = plt.colorbar(label="Temperature (Normalized)")

# Overlay streamlines to see how wind moves heat
plt.streamplot(X, Y, u, v, color='white', density=0.8, linewidth=0.5, arrowsize=0.5, zorder=2)

plt.title(f"Thermal Flow Past a {SHAPE_TYPE.capitalize()}")
plt.xlabel("Length (m)")
plt.ylabel("Height (m)")
plt.axis("equal")

# Draw the shape for reference
# Convert to numpy array and ensure it's closed
shape_points_array = np.array(shape_points)
shape_patch = plt.Polygon(shape_points_array, facecolor='black', edgecolor='white', linewidth=2, zorder=10)
plt.gca().add_patch(shape_patch)

plt.savefig(f"flow/{SHAPE_TYPE}_thermal_flow.png", dpi=150, bbox_inches='tight')
print(f"Saved thermal flow visualization to flow/{SHAPE_TYPE}_thermal_flow.png")