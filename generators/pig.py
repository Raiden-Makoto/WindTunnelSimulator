import numpy as np #type: ignore
import deepxde as dde #type: ignore

def generate_flying_pig():
    """
    Defines the coordinates for a flying pig.
    CORRECTED: Now facing LEFT (into the wind).
    """
    # I flipped the X values around the center axis (x=0.5)
    pig_points = np.array([
        [0.38, 0.48], # Snout Tip (Now on Left)
        [0.38, 0.52], # Snout Top
        [0.42, 0.55], # Forehead
        [0.44, 0.62], # Ear Tip
        [0.46, 0.58], # Ear Base
        [0.60, 0.58], # Back
        [0.65, 0.55], # Rump
        [0.67, 0.50], # Tail Base
        [0.65, 0.40], # Rear Hoof
        [0.60, 0.42], # Rear Leg Top
        [0.52, 0.40], # Belly
        [0.48, 0.42], # Front Leg Top
        [0.45, 0.38], # Front Hoof
        [0.42, 0.45], # Neck/Jaw
    ])
    
    # Create DeepXDE Polygon
    geom = dde.geometry.Polygon(pig_points)
    
    return geom, pig_points