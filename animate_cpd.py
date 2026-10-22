import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection

# ============================================================
# LOAD HISTORY
# ============================================================

data = np.load(
    "source/history.npz",
    allow_pickle=True
)

pos_history = data["pos"]
tri_history = data["tri"]
pre_history = data["pre"]
for_history = data["forces"]
btri_history = data["btri"]

print("Available keys :", data.files)

print("pos_history shape    :", pos_history.shape)
print("forces_history shape :", for_history.shape)
print("tri frames           :", len(tri_history))
print("pressure frames      :", len(pre_history))
print("btri frames          :", len(btri_history))

n_frames = len(pos_history)

# ============================================================
# PRINT SAMPLE INFORMATION
# ============================================================

print("\nExample frame information:")

for i in range(min(5, n_frames)):

    tri_i = np.asarray(tri_history[i])
    pre_i = np.asarray(pre_history[i])
    btri_i = np.asarray(btri_history[i])

    print(
        f"Frame {i}: "
        f"tri = {tri_i.shape}, "
        f"pressure = {pre_i.shape}, "
        f"btri = {btri_i.shape}"
    )

# ============================================================
# CHECK POSITION / FORCE CONSISTENCY
# ============================================================

if pos_history.shape != for_history.shape:

    raise ValueError(
        "Position and force history must have the same shape.\n"
        f"Positions : {pos_history.shape}\n"
        f"Forces    : {for_history.shape}"
    )

# ============================================================
# PRESSURE RANGE
# ============================================================

# Fixed pressure range
pressure_min = 0
pressure_max = 2

if pressure_min == pressure_max:
    pressure_max = pressure_min + 1.0

print("\nPressure colour range:")
print("Minimum pressure =", pressure_min)
print("Maximum pressure =", pressure_max)

# ============================================================
# FORCE MAGNITUDE
# ============================================================

force_magnitude = np.linalg.norm(
    for_history,
    axis=2
)

if np.all(np.isnan(force_magnitude)):

    global_max_force = 0.0

else:

    global_max_force = np.nanmax(
        force_magnitude
    )

print("\nMaximum force magnitude =", global_max_force)

# ============================================================
# GLOBAL PLOT LIMITS
# ============================================================

x_min = np.nanmin(
    pos_history[:, :, 0]
)

x_max = np.nanmax(
    pos_history[:, :, 0]
)

y_min = np.nanmin(
    pos_history[:, :, 1]
)

y_max = np.nanmax(
    pos_history[:, :, 1]
)

dx = x_max - x_min
dy = y_max - y_min

if dx == 0:
    dx = 1.0

if dy == 0:
    dy = 1.0

geometry_size = max(
    dx,
    dy
)

margin = 0.05

# ============================================================
# SAME-LENGTH FORCE GLYPHS
# ============================================================

# All non-zero force arrows will have exactly
# the same DISPLAY length.

arrow_length = 0.04 * geometry_size

# Avoid division by zero
safe_magnitude = np.where(
    force_magnitude > 0.0,
    force_magnitude,
    1.0
)

# Convert force vector into unit direction vector
unit_forces = (
    for_history /
    safe_magnitude[:, :, None]
)

# Give every non-zero force vector identical length
scaled_forces = (
    unit_forces *
    arrow_length
)

# Zero force --> no arrow
scaled_forces[
    force_magnitude == 0.0
] = 0.0

# Remove NaN / inf
scaled_forces = np.nan_to_num(
    scaled_forces,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# ============================================================
# FUNCTION: TRIANGLE VERTICES
# ============================================================

def get_triangle_vertices(
    pos,
    tri
):

    tri = np.asarray(
        tri,
        dtype=np.int64
    )

    if tri.size == 0:

        return np.empty(
            (0, 3, 2),
            dtype=float
        )

    tri = tri.reshape(
        -1,
        3
    )

    return pos[tri]

# ============================================================
# FUNCTION: FIND BOUNDARY TRIANGLES
# ============================================================

def get_boundary_information(
    tri,
    btri
):

    """
    Returns
    -------

    boundary_mask :
        Boolean array with length = number of triangles.

    boundary_tri :
        Triangle connectivity corresponding to boundary triangles.


    This function supports two common btri formats:

    1. btri contains triangle connectivity:
           [[n0, n1, n2],
            [n3, n4, n5],
            ...]

    2. btri contains row indices of tri:
           [0, 5, 10, 14, ...]
    """

    tri = np.asarray(
        tri,
        dtype=np.int64
    ).reshape(-1, 3)

    btri = np.asarray(
        btri
    )

    n_tri = len(tri)

    boundary_mask = np.zeros(
        n_tri,
        dtype=bool
    )

    # --------------------------------------------------------
    # No boundary triangles
    # --------------------------------------------------------

    if btri.size == 0:

        return (
            boundary_mask,
            np.empty(
                (0, 3),
                dtype=np.int64
            )
        )

    # --------------------------------------------------------
    # CASE 1:
    # btri contains explicit triangle connectivity
    # --------------------------------------------------------

    if (
        btri.ndim == 2
        and
        btri.shape[1] == 3
    ):

        boundary_tri = np.asarray(
            btri,
            dtype=np.int64
        ).reshape(-1, 3)

        # Compare triangles independent of node ordering.
        #
        # Example:
        # [1, 5, 3]
        # and
        # [3, 1, 5]
        #
        # are considered the same triangle.

        tri_sorted = np.sort(
            tri,
            axis=1
        )

        boundary_sorted = np.sort(
            boundary_tri,
            axis=1
        )

        # Build set for quick lookup

        boundary_set = {
            tuple(row)
            for row in boundary_sorted
        }

        boundary_mask = np.array(
            [
                tuple(row) in boundary_set
                for row in tri_sorted
            ],
            dtype=bool
        )

        # Important:
        # Use current tri connectivity rather than directly
        # using btri, so it always corresponds to current mesh.

        boundary_tri = tri[
            boundary_mask
        ]

        return (
            boundary_mask,
            boundary_tri
        )

    # --------------------------------------------------------
    # CASE 2:
    # btri contains triangle indices
    # --------------------------------------------------------

    boundary_indices = np.asarray(
        btri,
        dtype=np.int64
    ).reshape(-1)

    # Remove invalid indices
    boundary_indices = boundary_indices[
        (boundary_indices >= 0)
        &
        (boundary_indices < n_tri)
    ]

    boundary_mask[
        boundary_indices
    ] = True

    boundary_tri = tri[
        boundary_mask
    ]

    return (
        boundary_mask,
        boundary_tri
    )

# ============================================================
# INITIAL FRAME
# ============================================================

pos0 = np.asarray(
    pos_history[0],
    dtype=float
)

tri0 = np.asarray(
    tri_history[0],
    dtype=np.int64
).reshape(-1, 3)

pressure0 = np.asarray(
    pre_history[0],
    dtype=float
).reshape(-1)

btri0 = np.asarray(
    btri_history[0]
)

# ============================================================
# CHECK PRESSURE / TRIANGLE MATCH
# ============================================================

if len(tri0) != len(pressure0):

    raise ValueError(
        f"Frame 0 has {len(tri0)} triangles "
        f"but {len(pressure0)} pressure values."
    )

# ============================================================
# IDENTIFY BOUNDARY TRIANGLES
# ============================================================

boundary_mask0, boundary_tri0 = (
    get_boundary_information(
        tri0,
        btri0
    )
)

# ============================================================
# SEPARATE INTERIOR TRIANGLES
# ============================================================

interior_mask0 = ~boundary_mask0

interior_tri0 = tri0[
    interior_mask0
]

interior_pressure0 = pressure0[
    interior_mask0
]

# ============================================================
# CREATE TRIANGLE VERTICES
# ============================================================

interior_vertices0 = get_triangle_vertices(
    pos0,
    interior_tri0
)

boundary_vertices0 = get_triangle_vertices(
    pos0,
    boundary_tri0
)

# ============================================================
# FORCE INITIAL FRAME
# ============================================================

force0 = scaled_forces[0]

force_mag0 = force_magnitude[0]

# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 8)
)

ax.set_aspect(
    "equal",
    "box"
)

ax.set_xlim(
    x_min - margin * dx,
    x_max + margin * dx
)

ax.set_ylim(
    y_min - margin * dy,
    y_max + margin * dy
)

ax.set_xlabel("X")
ax.set_ylabel("Y")

# ============================================================
# PRESSURE-COLOURED INTERIOR TRIANGLES
# ============================================================

pressure_mesh = PolyCollection(

    interior_vertices0,

    array=interior_pressure0,

    cmap="jet",

    edgecolors="black",

    linewidths=0.35,

    zorder=1
)

pressure_mesh.set_clim(
    pressure_min,
    pressure_max
)

ax.add_collection(
    pressure_mesh
)

# ============================================================
# BLACK BOUNDARY TRIANGLES
# ============================================================

boundary_mesh = PolyCollection(

    boundary_vertices0,

    facecolors="black",

    edgecolors="black",

    linewidths=0.5,

    zorder=2
)

ax.add_collection(
    boundary_mesh
)

# ============================================================
# PARTICLES
# ============================================================

scat = ax.scatter(

    pos0[:, 0],
    pos0[:, 1],

    s=5,

    c="black",

    zorder=4
)

# ============================================================
# FORCE GLYPHS
# ============================================================

# quiver = ax.quiver(

#     pos0[:, 0],
#     pos0[:, 1],

#     force0[:, 0],
#     force0[:, 1],

#     force_mag0,

#     angles="xy",

#     scale_units="xy",

#     scale=1,

#     cmap="viridis",

#     width=0.0015,

#     headwidth=4,

#     headlength=5,

#     headaxislength=4.5,

#     pivot="tail",

#     zorder=5
# )

# Same force colour scale for every frame

# quiver.set_clim(
#     0.0,
#     global_max_force
#     if global_max_force > 0.0
#     else 1.0
# )

# ============================================================
# PRESSURE COLORBAR
# ============================================================

cbar_pressure = fig.colorbar(

    pressure_mesh,

    ax=ax,

    fraction=0.046,

    pad=0.04
)

cbar_pressure.set_label(
    "Pressure"
)

# ============================================================
# FORCE COLORBAR
# ============================================================

# cbar_force = fig.colorbar(

#     # quiver,

#     ax=ax,

#     fraction=0.046,

#     pad=0.10
# )

# cbar_force.set_label(
#     "Force magnitude"
# )

# ============================================================
# TITLE
# ============================================================

title = ax.set_title("")

# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(frame):

    # ========================================================
    # CURRENT POSITIONS
    # ========================================================

    pos = np.asarray(
        pos_history[frame],
        dtype=float
    )

    # ========================================================
    # CURRENT TRIANGLES
    # ========================================================

    tri = np.asarray(
        tri_history[frame],
        dtype=np.int64
    ).reshape(-1, 3)

    # ========================================================
    # CURRENT PRESSURE
    # ========================================================

    pressure = np.asarray(
        pre_history[frame],
        dtype=float
    ).reshape(-1)

    # ========================================================
    # CURRENT BOUNDARY TRIANGLES
    # ========================================================

    btri = np.asarray(
        btri_history[frame]
    )

    # ========================================================
    # CHECK TRIANGLE / PRESSURE MATCH
    # ========================================================

    if len(tri) != len(pressure):

        raise ValueError(
            f"Frame {frame}: "
            f"{len(tri)} triangles but "
            f"{len(pressure)} pressure values."
        )

    # ========================================================
    # IDENTIFY BOUNDARY TRIANGLES
    # ========================================================

    boundary_mask, boundary_tri = (
        get_boundary_information(
            tri,
            btri
        )
    )

    # ========================================================
    # INTERIOR TRIANGLES
    # ========================================================

    interior_mask = ~boundary_mask

    interior_tri = tri[
        interior_mask
    ]

    interior_pressure = pressure[
        interior_mask
    ]

    # ========================================================
    # TRIANGLE GEOMETRY
    # ========================================================

    interior_vertices = get_triangle_vertices(
        pos,
        interior_tri
    )

    boundary_vertices = get_triangle_vertices(
        pos,
        boundary_tri
    )

    # ========================================================
    # UPDATE PRESSURE-COLOURED TRIANGLES
    # ========================================================

    pressure_mesh.set_verts(
        interior_vertices
    )

    pressure_mesh.set_array(
        interior_pressure
    )

    # ========================================================
    # UPDATE BLACK BOUNDARY TRIANGLES
    # ========================================================

    boundary_mesh.set_verts(
        boundary_vertices
    )

    # ========================================================
    # UPDATE PARTICLES
    # ========================================================

    scat.set_offsets(
        pos
    )

    # ========================================================
    # CURRENT FORCE
    # ========================================================

    # force = scaled_forces[
    #     frame
    # ]

    # force_mag = force_magnitude[
    #     frame
    # ]

    # # ========================================================
    # # UPDATE FORCE POSITIONS
    # # ========================================================

    # quiver.set_offsets(
    #     pos
    # )

    # # ========================================================
    # # UPDATE FORCE DIRECTIONS AND COLOURS
    # # ========================================================

    # quiver.set_UVC(

    #     force[:, 0],

    #     force[:, 1],

    #     force_mag
    # )

    # ========================================================
    # STATISTICS
    # ========================================================

    if pressure.size > 0:

        current_pressure_min = np.nanmin(
            pressure
        )

        current_pressure_max = np.nanmax(
            pressure
        )

    else:

        current_pressure_min = 0.0
        current_pressure_max = 0.0


    # if force_mag.size > 0:

    #     current_force_max = np.nanmax(
    #         force_mag
    #     )

    # else:

    #     current_force_max = 0.0

    # ========================================================
    # TITLE
    # ========================================================

    # title.set_text(

    #     f"Frame {frame + 1}/{n_frames}    "

    #     f"Triangles = {len(tri)}    "

    #     f"Boundary = {np.sum(boundary_mask)}    "

    #     f"P = [{current_pressure_min:.3e}, "
    #     f"{current_pressure_max:.3e}]    "

    #     f"Max Force = {current_force_max:.3e}"
    # )

    # return (
    #     pressure_mesh,
    #     boundary_mesh,
    #     scat,
    #     quiver,
    #     title
    # )

# ============================================================
# ANIMATION
# ============================================================

ani = FuncAnimation(

    fig,

    update,

    frames=n_frames,

    interval=500,

    blit=False,

    repeat=True
)

plt.tight_layout()

plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Load saved positions
# #pos_history = np.load("source/pos_history.npy")
# data = np.load("source/history.npz", allow_pickle=True)
# pos_history = data["pos"]
# n_snaps, n_nodes, _ = pos_history.shape

# fig, ax = plt.subplots()
# scat = ax.scatter([], [], s=10)
# ax.set_aspect('equal', 'box')

# # Add margin
# x_min, x_max = pos_history[:,:,0].min(), pos_history[:,:,0].max()
# y_min, y_max = pos_history[:,:,1].min(), pos_history[:,:,1].max()
# margin = 0.05
# ax.set_xlim(x_min - margin*(x_max-x_min), x_max + margin*(x_max-x_min))
# ax.set_ylim(y_min - margin*(y_max-y_min), y_max + margin*(y_max-y_min))

# title = ax.set_title("")

# def update(frame):
#     scat.set_offsets(pos_history[frame])
#     title.set_text(f"Step {frame+1}/{n_snaps}")
#     return scat, title

# # Important: blit=False in desktop script
# ani = FuncAnimation(fig, update, frames=n_snaps, interval=20, blit=False)

# plt.show()
