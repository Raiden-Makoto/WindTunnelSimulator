import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde #type: ignore
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore

# ==========================================
# 1. SETUP & PHYSICS CONSTANTS
# ==========================================
# Rayleigh Number (Ra) controls how turbulent it is.
# Ra = 10,000 is enough to get nice circular rolls.
Ra = 10000 
Pr = 1.0   # Prandtl Number (Fluid property)

# Domain: A rectangular box (2 wide, 1 tall)
geom = dde.geometry.Rectangle([0, 0], [2, 1])

# ==========================================
# 2. THE CONVECTION EQUATIONS
# ==========================================
def pde(x, y):
    # y output: [u, v, p, T]
    u, v, p, T = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    
    # First derivatives
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    dT_x = dde.grad.jacobian(y, x, i=3, j=0)
    dT_y = dde.grad.jacobian(y, x, i=3, j=1)
    
    # Second derivatives
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    dT_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
    dT_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)
    
    # 1. Momentum X (Standard Navier-Stokes)
    # (u*du/dx + v*du/dy) = -dp/dx + Pr * (d2u/dx2 + d2u/dy2)
    mom_u = (u * du_x + v * du_y) + dp_x - Pr * (du_xx + du_yy)
    
    # 2. Momentum Y (WITH BUOYANCY)
    # The "+ Ra * Pr * T" term is the magic. Heat (T) creates upward force.
    mom_v = (u * dv_x + v * dv_y) + dp_y - Pr * (dv_xx + dv_yy) - (Ra * Pr * T)
    
    # 3. Continuity (Mass Conservation)
    cont = du_x + dv_y
    
    # 4. Energy Equation (Heat moving with fluid)
    energy = (u * dT_x + v * dT_y) - (dT_xx + dT_yy)
    
    return [mom_u, mom_v, cont, energy]

# ==========================================
# 3. BOUNDARY CONDITIONS
# ==========================================
def boundary_bottom(x, on_boundary): return on_boundary and np.isclose(x[1], 0)
def boundary_top(x, on_boundary):    return on_boundary and np.isclose(x[1], 1)
def boundary_sides(x, on_boundary):  return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], 2))

# Walls are slippery (Free Slip) or sticky (No Slip)? Let's do No-Slip.
bc_wall_u = dde.icbc.DirichletBC(geom, lambda x: 0, lambda x, on_b: on_b, component=0)
bc_wall_v = dde.icbc.DirichletBC(geom, lambda x: 0, lambda x, on_b: on_b, component=1)

# HEAT: Bottom is HOT (1), Top is COLD (0)
bc_temp_bot = dde.icbc.DirichletBC(geom, lambda x: 1, boundary_bottom, component=3)
bc_temp_top = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=3)

# Insulation on sides (Heat doesn't escape left/right)
# (Technically should be Neumann BC, but Dirichlet 0.5 is a stable approximation for this demo)
# Let's try Linear Gradient on sides to help it converge: T = 1 - y
bc_temp_sides = dde.icbc.DirichletBC(geom, lambda x: 1 - x[:, 1:2], boundary_sides, component=3)

bcs = [bc_wall_u, bc_wall_v, bc_temp_bot, bc_temp_top, bc_temp_sides]

# ==========================================
# 4. TRAINING
# ==========================================
data = dde.data.PDE(geom, pde, bcs, num_domain=2500, num_boundary=500, num_test=1000)
net = dde.nn.FNN([2] + [64] * 5 + [4], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
print("Heating the soup... (This trains the Convection Cells)")
model.train(iterations=10000)

# ==========================================
# 5. VISUALIZATION
# ==========================================
print("Plotting...")
x = np.linspace(0, 2, 200)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
xy = np.vstack((X.ravel(), Y.ravel())).T

uvp = model.predict(xy)
u = uvp[:, 0].reshape(X.shape)
v = uvp[:, 1].reshape(X.shape)
temp = uvp[:, 3].reshape(X.shape)

plt.figure(figsize=(10, 5))
# 1. Plot Temperature Background
plt.contourf(X, Y, temp, levels=100, cmap="RdBu_r") # Red = Hot, Blue = Cold
plt.colorbar(label="Temperature")

# 2. Plot Velocity Streamlines (The Cells)
plt.streamplot(X, Y, u, v, color='black', density=1.5, linewidth=0.6, arrowsize=0.8)

plt.title(f"Rayleigh-Bénard Convection (Ra={Ra})")
plt.xlabel("Width")
plt.ylabel("Height")
plt.savefig(f"convection/convection_{Ra}.png")