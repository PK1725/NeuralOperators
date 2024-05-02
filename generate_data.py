import os
import utils
import numpy as np

L = 2*np.pi
t_max = 1.6037 # Maximum simulation time
nu = 0.01 

# Simulation settings
n = 2**11
dt = 0.0001 #0.001  # Time step size

# Downsample the simulation for training
space_res = 2**7
time_res = 2**6

n_simulations = 1000  # Number of simulations
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

np.random.seed(69) # for training data
# np.random.seed(666) # for test data
x_grid,u_t_train, u_0_train = utils.simulate_IC(n_simulations,initial_conditions,L, n, t_max, dt, nu,
                                                plotting=False,keep_first_t=False,
                                                space_res=space_res,time_res=time_res,old_ic=False,solver='Fourier')

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




