import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from numba import njit, prange
import numba
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"

# List of cpu functions

def delaunay_triangulate(
    pos: NDArray[np.floating], edge_cutoff
):

    # --------------------------------------------------
    # Generate Delaunay triangulation
    # --------------------------------------------------

    tri = Delaunay(pos)
    tri_simplices = tri.simplices

    # --------------------------------------------------
    # Calculate triangle edge lengths
    # --------------------------------------------------

    v0 = pos[tri_simplices[:, 0]]
    v1 = pos[tri_simplices[:, 1]]
    v2 = pos[tri_simplices[:, 2]]

    edge_01_length = np.linalg.norm(
        v1 - v0,
        axis=1
    )

    edge_12_length = np.linalg.norm(
        v2 - v1,
        axis=1
    )

    edge_20_length = np.linalg.norm(
        v0 - v2,
        axis=1
    )

    # Maximum edge length of each triangle
    max_triangle_edge_length = np.maximum.reduce(
        (
            edge_01_length,
            edge_12_length,
            edge_20_length
        )
    )

    # --------------------------------------------------
    # Filter triangles based on edge length
    # --------------------------------------------------

    max_edge_length_threshold = edge_cutoff

    filtered_tri = tri_simplices[
        max_triangle_edge_length
        < max_edge_length_threshold
    ]

    # --------------------------------------------------
    # Find boundary triangles
    # --------------------------------------------------

    # Create the three edges of every filtered triangle
    edges = np.vstack((
        filtered_tri[:, [0, 1]],
        filtered_tri[:, [1, 2]],
        filtered_tri[:, [2, 0]]
    ))

    # Store every edge in ascending vertex order
    sorted_edges = np.sort(
        edges,
        axis=1
    )

    # Find unique edges and count how many triangles
    # share each edge
    unique_edges, inverse_indices, edge_counts = np.unique(
        sorted_edges,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    # An edge appearing only once is a boundary edge
    boundary_edges = unique_edges[
        edge_counts == 1
    ]

    # --------------------------------------------------
    # Find boundary triangles
    # --------------------------------------------------

    boundary_nodes = np.unique(
        boundary_edges
    )

    is_boundary_triangle = np.any(
        np.isin(
            filtered_tri,
            boundary_nodes
        ),
        axis=1
    )

    boundary_triangle_indices = np.where(
        is_boundary_triangle
    )[0]

    boundary_triangles = filtered_tri[
        boundary_triangle_indices
    ]

    # --------------------------------------------------
    # Calculate vertex_neighbor_vertices equivalent
    # --------------------------------------------------

    # Add edges in both directions because
    # vertex adjacency is symmetric

    row = np.concatenate((
        edges[:, 0],
        edges[:, 1]
    ))

    col = np.concatenate((
        edges[:, 1],
        edges[:, 0]
    ))

    adjacency = csr_matrix(
        (
            np.ones(
                len(row),
                dtype=np.uint8
            ),
            (
                row,
                col
            )
        ),
        shape=(
            len(pos),
            len(pos)
        )
    )

    adjacency.sum_duplicates()
    adjacency.data[:] = 1
    adjacency.sort_indices()

    # Equivalent to tri.vertex_neighbor_vertices
    indptr = adjacency.indptr
    indices = adjacency.indices


    return (
        filtered_tri,
        boundary_triangles,
        boundary_nodes,
        indptr
    )

import numpy as np


def calculate_node_angles(pos, filtered_tri):
    """
    Calculate total angle occupied by triangles around every point.

    Parameters
    ----------
    pos : (N, 2) ndarray
        Particle coordinates.

    filtered_tri : (NT, 3) ndarray
        Triangle connectivity.

    Returns
    -------
    node_angle_rad : (N,) ndarray
        Sum of connected triangle angles in radians.

    node_angle_deg : (N,) ndarray
        Sum of connected triangle angles in degrees.
    """

    n_nodes = len(pos)

    # Triangle vertices
    p0 = pos[filtered_tri[:, 0]]
    p1 = pos[filtered_tri[:, 1]]
    p2 = pos[filtered_tri[:, 2]]

    # --------------------------------------------------
    # Vectors at vertex 0
    # --------------------------------------------------

    a0 = p1 - p0
    b0 = p2 - p0

    # --------------------------------------------------
    # Vectors at vertex 1
    # --------------------------------------------------

    a1 = p0 - p1
    b1 = p2 - p1

    # --------------------------------------------------
    # Vectors at vertex 2
    # --------------------------------------------------

    a2 = p0 - p2
    b2 = p1 - p2

    # --------------------------------------------------
    # Function to calculate angle between vector pairs
    # --------------------------------------------------

    def vector_angle(a, b):

        dot = np.einsum(
            "ij,ij->i",
            a,
            b
        )

        cross = (
            a[:, 0] * b[:, 1]
            -
            a[:, 1] * b[:, 0]
        )

        # atan2 is numerically more robust than arccos
        angle = np.arctan2(
            np.abs(cross),
            dot
        )

        return angle

    # Corner angles of every triangle
    angle0 = vector_angle(a0, b0)
    angle1 = vector_angle(a1, b1)
    angle2 = vector_angle(a2, b2)

    # --------------------------------------------------
    # Accumulate triangle angles onto particles
    # --------------------------------------------------

    node_angle_rad = np.zeros(
        n_nodes,
        dtype=float
    )

    np.add.at(
        node_angle_rad,
        filtered_tri[:, 0],
        angle0
    )

    np.add.at(
        node_angle_rad,
        filtered_tri[:, 1],
        angle1
    )

    np.add.at(
        node_angle_rad,
        filtered_tri[:, 2],
        angle2
    )

    node_angle_deg = np.degrees(
        node_angle_rad
    )

    return node_angle_rad, node_angle_deg

def calc_ele_areas_cpu(pos:NDArray[np.floating], tri:NDArray[np.integer])->NDArray[np.floating]:
    p1 = pos[tri[:,0]]
    p2 = pos[tri[:,1]]
    p3 = pos[tri[:,2]]
    areas = 0.5*np.abs(p1[:,0]*(p2[:,1]-p3[:,1])+p2[:,0]*(p3[:,1]-p1[:,1])+p3[:,0]*(p1[:,1]-p2[:,1]))
    return areas

def calc_ele_densities_cpu(pos:NDArray[np.floating], tri:NDArray[np.integer],rho_node:NDArray[np.floating])->NDArray[np.floating]:
    r1 = rho_node[tri[:,0]]
    r2 = rho_node[tri[:,1]]
    r3 = rho_node[tri[:,2]]
    rho_ele = 0.33*(r1+r2+r3)
    return rho_ele

def calc_node_areas_cpu(pos:NDArray[np.floating], tri:NDArray[np.integer], areas:NDArray[np.floating])->NDArray[np.floating]:
    contribs = np.repeat(areas*0.33,3)
    flat_nodes = tri.ravel()
    return np.bincount(flat_nodes, weights = contribs, minlength = pos.shape[0])
#----------------------changes9/7
# def calc_ele_mass_cpu(node_mass,boundary_nodes,indptr,tri_simplices,node_angle_deg):

#     num_neighbors = np.diff(indptr)

#     m1 = node_mass/num_neighbors[tri_simplices[:,0]]*(node_angle_deg[tri_simplices[:,0]]/360)
#     m2 = node_mass/num_neighbors[tri_simplices[:,1]]*(node_angle_deg[tri_simplices[:,0]]/360)
#     m3 = node_mass/num_neighbors[tri_simplices[:,2]]*(node_angle_deg[tri_simplices[:,0]]/360)
#     return m1+m2+m3

@njit
def is_triangle_in_array(boundary_triangles, target_tri):
    # Iterates over each row manually
    for i in range(len(boundary_triangles)):
        # Compares all 3 elements of the triangle
        if (boundary_triangles[i, 0] == target_tri[0] and 
            boundary_triangles[i, 1] == target_tri[1] and 
            boundary_triangles[i, 2] == target_tri[2]):
            return True
    return False


@njit(cache=True, fastmath=True)
def calc_force_f_numba_cpu(pos:NDArray[np.floating], vel:NDArray[np.floating], tri:NDArray[np.integer],boundary_triangles: NDArray[np.integer],areas:NDArray[np.floating], node_mass:float, rho:float = 10, mu:float = 1.0, K:float = 1.0, rho_ele:float = 0.1, g:float =10)->NDArray[np.floating]:
    Forces = np.zeros((pos.shape[0],2))

    for idx,nodes in enumerate(tri):

        
        if (rho_ele[idx]-rho)<0: 
            coef =0

        else:
            coef = -K*(rho_ele[idx]**2-rho**2)           #////// K/A0[idx]*(2*areas[idx]-A0[idx]) ///////  K*(A0**2 * np.log(areas[idx]) - 1) 

        for m in range(3):
            n = (m+1)%3
            o = (m+2)%3
            n1, n2, n3 = nodes[m], nodes[n], nodes[o]
            xdi, ydi = pos[n1,0], pos[n1,1]
            xdj, ydj = pos[n2,0], pos[n2,1]
            xdk, ydk = pos[n3,0], pos[n3,1]
            
            vxi, vyi = vel[n1,0], vel[n1,1]
            vxj, vyj = vel[n2,0], vel[n2,1]
            vxk, vyk = vel[n3,0], vel[n3,1]
            
            rijx = (xdj-xdi)
            rijy = (ydj-ydi)
            rikx = (xdk-xdi)
            riky = (ydk-ydi)
            vijx = (vxj-vxi)
            vijy = (vyj-vyi)
            vikx = (vxk-vxi)
            viky = (vyk-vyi)
            #new formulation variables
            rjky = (ydk-ydj)
            rkjx = (xdj-xdk)
            

            fxp = -coef*(pos[n2,1]-pos[n3,1])
            fyp = coef*(pos[n2,0]-pos[n3,0])

            # Viscous force
            #fxv = -(2*mu/(0.00001+rijx*riky-rikx*rijy)**2)*(-2*riky*(vijx*riky-vijy*rikx)+2*rijy*(viky*rijx-vikx*rijy)+(rijy-riky)*(vikx*riky-viky*rikx+vijy*rijx-vijx*rijy))
            #fyv = -(2*mu/(0.00001+rijx*riky-rikx*rijy)**2)*(2*rikx*(vijx*riky-vijy*rikx)-2*rijx*(viky*rijx-vikx*rijy)+(rikx-rijx)*(vikx*riky-viky*rikx+vijy*rijx-vijx*rijy))
            #new formulation
            #print(abs(rijx*riky - rikx*rijy))


            if abs(rijx*riky-rikx*rijy) > 1e-8 :
                fxv = (2*mu*areas[idx]/((rijx*riky-rikx*rijy)**2))*(2*rjky*(vijx*riky-vikx*rijy)+rkjx*(-vijx*rikx+vikx*rijx+vijy*riky-viky*rijy))  #(-vijy*rikx+viky*rijx)**2      
                fyv = (2*mu*areas[idx]/((rijx*riky-rikx*rijy)**2))*(2*rkjx*(-vijy*rikx+viky*rijx)+rjky*(-vijx*rikx+vikx*rijx+vijy*riky-viky*rijy)) #+(vijx*riky-vikx*rijy)**2
            
            else:
                fxv = 0
                fyv = 0

            Forces[n1,0] +=  fxv + fxp
            Forces[n1,1] +=  fyv + fyp
        
    return Forces #, fxp, fyp, fxv, fyv

def update_cpu(pos:NDArray[np.floating], vel:NDArray[np.floating], inv_m:NDArray[np.floating], forces:NDArray[np.floating], fixed_bc:NDArray[np.integer], force_bc:NDArray[np.floating], vel_bc:NDArray[np.floating], prev_forces, dt:float = 1e-5, g:float = 0, curr_time:float = 0)->tuple[:NDArray[np.floating], :NDArray[np.floating]]:
    
    force_nodes = force_bc[:, 0].astype(int)
    vel_nodes = vel_bc[:, 0].astype(int)
    
    if force_bc.size>0:
        forces[force_nodes,0] = force_bc[:,1]
        forces[force_nodes,1] = force_bc[:,2]
    #prev_vel = vel
    
    vel += (forces)*inv_m*dt
    #vel += (forces+prev_forces)*inv_m*dt*0.5

    vel[:,1] += -g*dt

    if fixed_bc.size>0:
        vel[fixed_bc,0] = 0
        vel[fixed_bc,1] = 0
        
    if vel_bc.size>0:
        # if curr_time < 0:         
        #     vel[vel_nodes,0] = vel_bc[:,1]*(curr_time/1000)
        #     vel[vel_nodes,1] = vel_bc[:,2]*(curr_time/1000)
        # else:
        vel[vel_nodes,0] = vel_bc[:,1]
        vel[vel_nodes,1] = vel_bc[:,2]              

    # for i in range(pos.shape[0]) :
    #     if pos[i,0] > 1.001 :
    #         pos[i,0] = pos[i,0]-1-1/119
    pos += vel*dt    
    #pos += prev_vel*dt+0.5*(prev_forces*inv_m*dt*dt)
    return pos,vel
