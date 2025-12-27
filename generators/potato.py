import numpy as np #type: ignore
import deepxde as dde #type: ignore

def generate_random_potato(center=(0.5, 0.5), avg_radius=0.15):
    """
    Generates a random organic shape (a potato) using polar coordinates.
    """
    # 1. Create random angles (sorted so the line connects properly)
    num_points = np.random.randint(10, 20) # More points = smoother potato
    angles = np.sort(np.random.rand(num_points) * 2 * np.pi)
    
    # 2. Vary the radius (Organic lumpiness)
    # We use sine waves to make it smooth, not jagged
    base_radius = avg_radius
    variation = np.random.rand(num_points) * 0.08 - 0.04 # +/- 0.04 variation
    radii = base_radius + variation
    
    # 3. Convert Polar (angle, radius) -> Cartesian (x, y)
    points = []
    for a, r in zip(angles, radii):
        points.append([
            center[0] + r * np.cos(a),
            center[1] + r * np.sin(a)
        ])
    
    # 4. Return DeepXDE Polygon and points for visualization
    return dde.geometry.Polygon(points), points