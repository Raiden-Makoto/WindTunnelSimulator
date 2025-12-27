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
