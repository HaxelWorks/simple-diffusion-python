

# %% imports
from tqdm import tqdm
import numpy as np
from PIL import Image

# Simulation parameters
nx, ny = 400, 400  # grid size
nit = 10000  # number of iterations
grid = np.zeros((nx, ny), dtype=np.float64)


def heat_region(grid, x, y, radius, strength):
    """Apply a heat source to the grid."""
    for i in range(x - radius, x + radius):
        for j in range(y - radius, y + radius):
            if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                grid[i, j] = strength
    return grid


def diffusion_step(grid, omega):
    """Perform one step of the diffusion simulation.
    Returns the updated grid."""

    # Compute the diffusion term
    diffusion = 0.25 * (
        np.roll(grid, 1, 0)
        + np.roll(grid, -1, 0)
        + np.roll(grid, 1, 1)
        + np.roll(grid, -1, 1)
        - 4 * grid
    )
    # Update the grid
    grid += omega * diffusion
    return grid


def diffusion(grid, omega, nit):
    """Perform the diffusion simulation for nit iterations.
    Use PIL to save the result as a gif."""
    # Create the gif

    images = []
    for i in tqdm(range(nit)):
        grid = diffusion_step(grid, omega)
        # Save the result as a gif

        image = Image.fromarray(np.uint8(grid * 255))
        if i % 100 == 0:
            images.append(image)

    images[0].save("diffusion.gif", save_all=True, append_images=images[1:], loop=0)
    return grid


#%% Run the simulation
# Apply a heat source
heat_region(grid, 200, 200, 50, 1)
diffusion(grid, omega=1, nit=nit)
# %%
