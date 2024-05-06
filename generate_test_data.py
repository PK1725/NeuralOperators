import os
import utils
import numpy as np


# load in training data to get training resolution
data_path = "data/data1000_128_64"

metadata, u_0_train, u_t_train, x_grid = utils.load_data(data_path)
metadata = metadata.item()

L = metadata['L']
t_max = metadata['t_max']
nu = metadata['nu']
n = metadata['n']
dt = metadata['dt']
space_res = metadata['space_res']
time_res = metadata['time_res']

test_resolutions = [[space_res,time_res], [2**4,2**4],[2**5,2**5],[2**6,2**6],[2**7,2**7],[2**8,2**8],[2**9,2**9],[2**10,2**10],[2**11,2**11]]


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

for res in test_resolutions:
    np.random.seed(666)
    x_grid_test,u_t_test, u_0_test = utils.simulate_IC(n_simulations,initial_conditions,L, n, t_max, dt, nu,
                                                plotting=False,keep_first_t=False,
                                                space_res=res[0],time_res=res[1],old_ic=False,solver='Fourier')
    
    # Save matrices
    save_folder = data_path + f"/test_data/data{n_simulations}_{res[0]}_{res[1]}"
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "x_grid.npy"), x_grid_test)
    np.save(os.path.join(save_folder, "u_t_train.npy"), u_t_test)
    np.save(os.path.join(save_folder, "u_0_train.npy"), u_0_test)
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




