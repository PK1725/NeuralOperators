import os
import utils
import numpy as np

L = 2*np.pi
t_max = 1 # Maximum simulation time
nu = 0.1 

# Simulation settings
n = 200
dt = 0.001 #0.001  # Time step size

# Downsample the simulation for training
space_res = 80
time_res = 50

n_simulations = 100  # Number of simulations
initial_conditions = {
    'u_initial_1': utils.u_initial_1,
    'u_initial_2': utils.u_initial_2,
    'u_initial_3': utils.u_initial_3,
    'u_initial_4': utils.u_initial_4,
    'u_initial_5': utils.u_initial_5,
    'u_initial_6': utils.u_initial_6,
    'u_initial_7': utils.u_initial_7,
    'u_initial_8': utils.u_initial_8,
}

x_grid,u_t_train, u_0_train = utils.simulate_IC(n_simulations,initial_conditions,L, n, t_max, dt, nu,
                                                plotting=False,keep_first_t=False,
                                                space_res=space_res,time_res=time_res)

# Save matrices
save_folder = f"data/data{n_simulations}_{space_res}_{time_res}"
os.makedirs(save_folder, exist_ok=True)
np.save(os.path.join(save_folder, "x_grid.npy"), x_grid)
np.save(os.path.join(save_folder, "u_t_train.npy"), u_t_train)
np.save(os.path.join(save_folder, "u_0_train.npy"), u_0_train)

# Save metadata
metadata = {
    'L': L,
    't_max': t_max,
    'nu': nu,
    'n': n,
    'dt': dt,
    'space_res': space_res,
    'time_res': time_res
}

np.save(os.path.join(save_folder, "metadata.npy"), metadata)




