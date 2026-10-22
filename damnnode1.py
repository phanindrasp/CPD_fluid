import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Geometry ──────────────────────────────────────────────
length = 0.2357 
height = 0.2357  

nx = 10
ny = 10

# Wall thickness (in nodes)
wall_left   = 2
wall_right  = 2
wall_bottom = 2

interior_cols = nx - wall_left - wall_right
interior_rows = ny - wall_bottom

fluid_cols = interior_cols // 1
fluid_rows = interior_rows // 2

# ── Coordinate arrays ─────────────────────────────────────
x_values = np.linspace(0, length, nx)
y_values = np.linspace(0, height, ny)

# ── Node lists ────────────────────────────────────────────
nodes       = []
fixed_nodes = []

# ── Helper functions ──────────────────────────────────────
def in_domain(i, j):
    in_left_wall  = (i < wall_left)
    in_right_wall = (i >= nx - wall_right)
    in_bottom     = (j < wall_bottom)
    in_fluid      = (wall_left <= i < wall_left + fluid_cols) and \
                    (wall_bottom <= j < wall_bottom + fluid_rows)
    return in_left_wall or in_right_wall or in_bottom or in_fluid

def is_fixed(i, j):
    return (i < wall_left) or (i >= nx - wall_right) or (j < wall_bottom)

# ── Ensure setup folder exists ────────────────────────────
os.makedirs("setup", exist_ok=True)

# ── Write setup/nodes.csv ─────────────────────────────────
with open("setup/nodes.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    for j, y in enumerate(y_values):
        for i, x in enumerate(x_values):
            if in_domain(i, j):
                nodes.append([x, y])
                writer.writerow([x, y])

nodes = np.array(nodes)

# ── Write setup/fixed.csv ─────────────────────────────────
with open("setup/fixed.csv", mode="w", newline="") as file:
    writer  = csv.writer(file)
    node_id = 0
    for j in range(ny):
        for i in range(nx):
            if in_domain(i, j):
                if is_fixed(i, j):
                    fixed_nodes.append(node_id)
                    writer.writerow([node_id])
                node_id += 1

fixed_nodes = np.array(fixed_nodes)

# ── Write setup/velocity.csv (empty — fluid starts at rest) 
#with open("setup/velocity.csv", mode="w", newline="") as file:
 #   pass
    

print("setup/nodes.csv, setup/fixed.csv & setup/velocity.csv created successfully!")
print(f"Total nodes  : {len(nodes)}")
print(f"Fixed nodes  : {len(fixed_nodes)}")
print(f"Fluid nodes  : {len(nodes) - len(fixed_nodes)}")
print(f"Wall inner bottom y : {wall_bottom * (height/(ny-1)):.6f} m")
print(f"Wall inner left  x  : {wall_left  * (length/(nx-1)):.6f} m")
print(f"Wall inner right x  : {length - wall_right * (length/(nx-1)):.6f} m")

# ── Plot ──────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.scatter(nodes[:, 0], nodes[:, 1], s=8, color='steelblue', label='All nodes')
plt.scatter(nodes[fixed_nodes, 0], nodes[fixed_nodes, 1], s=8, color='red', label='Fixed wall nodes')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Dam break mesh")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
