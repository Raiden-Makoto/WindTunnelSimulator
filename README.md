# WindTunnelSimulator

A Physics-Informed Neural Network (PINN) project for simulating 2D fluid flow and heat transfer in wind tunnels using DeepXDE.

## Overview

This project uses neural networks trained on physics equations to simulate:
- **Fluid flow** around objects (Navier-Stokes equations)
- **Heat transfer** via convection and diffusion
- **Thermal wakes** from hot objects in cold air streams

## Features

- 🌀 **Wind Tunnel Simulation**: Simulate flow past various shapes with configurable physics
- 🔥 **Thermal Flow**: Combined fluid dynamics and heat transfer
- 🌊 **Convection Simulation**: Rayleigh-Bénard convection patterns
- 🎨 **Multiple Shapes**: Support for different object geometries (potato, pig, etc.)
- 📊 **Visualization**: Automatic generation of flow and temperature plots

## Usage

### Wind Tunnel Simulation

Run the wind tunnel simulator with thermal flow:

```bash
python simulators/windtunnel.py
```

**Configuration** (in `windtunnel.py`):
- Change `SHAPE_TYPE` to switch between shapes: `"potato"` or `"pig"`
- Adjust physics constants:
  - `rho`: Density (default: 1.0)
  - `mu`: Viscosity (default: 0.002)
  - `u_inlet`: Inlet velocity (default: 1.0)
  - `thermal_diff`: Thermal diffusivity (default: 0.02)

**Output**: Saves thermal flow visualization to `flow/{shape}_thermal_flow.png`

### Convection Simulation

Run the Rayleigh-Bénard convection simulator:

```bash
python simulators/convection.py
```

**Configuration** (in `convection.py`):
- `Ra`: Rayleigh number (controls turbulence, default: 10000)
- `Pr`: Prandtl number (default: 1.0)

**Output**: Saves convection visualization to `convection/convection_{Ra}.png`

## Physics

### Wind Tunnel Simulator

Solves the coupled Navier-Stokes and heat equations:

- **Momentum equations**: Fluid flow with viscosity
- **Continuity equation**: Mass conservation
- **Heat equation**: Convective and diffusive heat transfer

**Boundary Conditions**:
- Inlet: Parabolic velocity profile, cold temperature (T=0)
- Outlet: Zero pressure
- Walls: No-slip, insulated
- Object: No-slip, hot (T=1)

### Convection Simulator

Models Rayleigh-Bénard convection:
- Hot bottom, cold top
- Natural convection driven by buoyancy
- Forms characteristic roll patterns

## Examples

The simulator generates visualizations showing:
- **Temperature fields**: Color-coded heat distribution
- **Streamlines**: Flow direction and patterns
- **Velocity fields**: Flow magnitude and direction
