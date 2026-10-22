import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from numba import njit
import numba
import time
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from cpu_utils import delaunay_triangulate,calculate_node_angles,calc_ele_areas_cpu,calc_ele_densities_cpu,calc_node_areas_cpu,calc_force_f_numba_cpu, update_cpu

nthreads=8
numba.set_num_threads(nthreads)   # or try 4/8/12; not always max is best

folder = Path(".")  # current folder
for file in folder.glob("*.npy"):
    file.unlink()
    print(f"Deleted {file}")

with open("config.yml","r") as f:
    config = yaml.safe_load(f)


N = config['simulation']['total_steps']
dt = float(config['simulation']['time_step'])
device = config['simulation']['device']
material = config['material']['which material'] 
g = config['simulation']['g'] 
write_steps = int(config['simulation']['write_every_%_steps']*N/100)
curr_time = 0

pos = np.loadtxt(config['geometry']['nodes'], delimiter = ',')

try:
    fixed_bc = np.loadtxt(config['geometry']['fixed'], delimiter=',', dtype=np.int32)
except (OSError, ValueError):
    fixed_bc = np.empty((0, 1))

try:
    force_bc = np.loadtxt(config['geometry']['force'], delimiter=',')
    force_bc[:, 0] = force_bc[:, 0].astype(np.int32)
except (OSError, ValueError):
    force_bc = np.empty((0, 3))
    
try:
    vel_bc = np.loadtxt(config['geometry']['velocity'], delimiter=',')
except (OSError, ValueError):
    vel_bc = np.empty((0, 3))

vel = np.zeros((pos.shape[0],2))
forces = np.zeros((pos.shape[0],2))
prev_forces = np.zeros((pos.shape[0],2))
ref_pos = np.asarray(pos.copy())
tls = np.empty((nthreads, pos.shape[0],2))



write_steps = max(1, write_steps)
n_snaps = N // write_steps 
pos_history = np.zeros((n_snaps, *pos.shape))
tri_history = []
btri_history = []
pre_history = []
for_history = np.zeros((n_snaps, *forces.shape))

x=0


mu = config['material']['constitutive_law']['fluid']['properties']['mu']
rho = config['material']['constitutive_law']['fluid']['properties']['rho']
K = config['material']['constitutive_law']['fluid']['properties']['K']
edge_cutoff = config['material']['constitutive_law']['fluid']['properties']['edge']
tri,boundary_tri,boundary_nodes,indptr  = delaunay_triangulate(pos, edge_cutoff)
ini_areas = calc_ele_areas_cpu(pos,tri)
node_mass = rho*np.sum(ini_areas)/pos.shape[0]/0.8
start = time.perf_counter() 


inv_m = (1.0/node_mass).reshape(-1,1) 

for curr_time in range(N):
    if curr_time%1000==0 :
        tri,boundary_tri,boundary_nodes,indptr = delaunay_triangulate(pos, edge_cutoff)

    node_angle_rad, node_angle_deg = calculate_node_angles(pos,tri)
    node_mass = node_mass*(node_angle_deg/360.0)
    area_ele = calc_ele_areas_cpu(pos,tri)
    node_area = calc_node_areas_cpu(pos,tri,area_ele)
    rho_node = rho*np.ones_like(node_mass, dtype=float)
    mask = node_area > 1e-8
    rho_node[mask] = node_mass[mask] / node_area[mask]
    rho_ele = calc_ele_densities_cpu(pos,tri,rho_node)
    
    #mass_ele = calc_ele_mass_cpu(node_mass,boundary_nodes,indptr,tri,node_angle_deg)
    #rho_ele = mass_ele/area_ele
    #pre = K*(rho_ele-rho)**2
    pre = rho_ele
    #prev_forces = forces     
    forces = calc_force_f_numba_cpu(pos,vel,tri,boundary_tri,area_ele,node_mass,rho,mu,K,rho_ele,g) 
    pos, vel = update_cpu(pos,vel,inv_m,forces,fixed_bc,force_bc,vel_bc,prev_forces,dt,g,curr_time)

    if (curr_time%write_steps==0):
        end = time.perf_counter()
        print(f"Time steps completed:", curr_time)
        print(f"Time elapsed (s):", end-start)
        pos_history[x] = pos
        btri_history.append(boundary_tri)
        tri_history.append(tri) 
        pre_history.append(pre) 
        for_history[x] = forces
        x += 1


np.savez("source/history.npz",pos=pos_history,forces=for_history,tri=np.array(tri_history, dtype=object),btri=np.array(btri_history, dtype=object),pre=np.array(pre_history, dtype=object))

#np.save("source/force_history.npy", force_history)
