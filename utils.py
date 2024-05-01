import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch,sys
from neuralop.models import TFNO
from neuralop import Trainer
import neuralop.training.callbacks as callbacks
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor

# Define the initial conditions

# sin 1
def u_initial_1(L, n):
    x = np.linspace(0, L, n)
    return np.sin(2 * np.pi * x / L)
# cos 1
def u_initial_2(L, n):
    x = np.linspace(0, L, n)
    return np.cos(2 * np.pi * x / L)
# bell
def u_initial_3(L, n):
    x = np.linspace(0, L, n)
    return np.exp(-((x - L / 2) ** 2) / 2)
# sin 2
def u_initial_4(L, n):
    x = np.linspace(0, L, n)
    return np.sin(4 * np.pi * x / L)
# cos 2
def u_initial_5(L, n):
    x = np.linspace(0, L, n)
    return np.cos(4 * np.pi * x / L)
# const
def u_initial_6(L, n):
    return np.ones(n)
# sin 3
def u_initial_7(L, n):
    x = np.linspace(0, L, n)
    return np.sin(6 * np.pi * x / L)
# cos 3
def u_initial_8(L, n):
    x = np.linspace(0, L, n)
    return np.cos(6 * np.pi * x / L)

def burgers_equation_simulation(u_initial, x_grid, dt, t_max, nu):
    """
    Simulate data from Burger's equation using finite difference method.
    
    Parameters:
        u_initial (array): Initial condition of u(x,0).
        x_grid (array): Array representing spatial grid points.
        dt (float): Time step size.
        t_max (float): Maximum simulation time.
        nu (float): Diffusion coefficient.
        
    Returns:
        u_solution (2D array): Solution array with shape (len(x_grid), num_steps).
        t_points (array): Array of time points.
    """
    tolerance = 1e-10  # Define a tolerance level for comparison
    if not np.isclose(u_initial[0], u_initial[-1], atol=tolerance):
        raise ValueError('u(0,0) is not approximately equal to u(L,0). Please select another u_initial.')

    num_steps = int(t_max / dt) + 1
    t_points = np.linspace(0, t_max, num_steps)
    dx = x_grid[1] - x_grid[0] # L/n
    u_solution = np.zeros((len(x_grid), num_steps))
    
    # Set initial condition
    u_solution[:, 0] = u_initial
    
    for i in range(1, num_steps):
        u_prev = u_solution[:, i - 1]
        u_next = np.zeros_like(u_prev)
        
        for j in range(1, len(x_grid) - 1):
            u_next[j] = u_prev[j] - u_prev[j] * (dt / dx) * (u_prev[j] - u_prev[j - 1]) + (nu * dt / dx**2) * (u_prev[j + 1] - 2 * u_prev[j] + u_prev[j - 1])
        
        # Boundary conditions (periodic boundary)
        u_next[0] = u_next[-2]
        u_next[-1] = u_next[1]
        
        u_solution[:, i] = u_next
    
    return u_solution, t_points


def burgers_equation_simulation2(u_initial, x_grid, dt, t_max, nu,space_res,time_res,keep_first_t=False):
    u_solution, t_points = burgers_equation_simulation(u_initial, x_grid, dt, t_max, nu)
    time_res_offset = int(int(t_max/dt)/time_res)
    space_res_offset = int(len(x_grid)/space_res)

    #downsample
    t_points = t_points[::time_res_offset]
    u_initial = u_initial[::space_res_offset]
    u_solution = u_solution[::space_res_offset,::time_res_offset]
    x_grid = x_grid[::space_res_offset]
    if not keep_first_t:
        u_solution = u_solution[:,1:]
        t_points = t_points[1:]
    return x_grid,t_points,u_initial,u_solution

 #Define IC from fourier 
def u_initial_const(n_xpoints):
    return np.ones(n_xpoints)

def u_initial_cos(n_xpoints,n_freq):
    x = np.linspace(0, 2*np.pi, n_xpoints)
    return np.cos(n_freq * x)

def u_initial_sin(n_xpoints,n_freq):
    x = np.linspace(0, 2*np.pi, n_xpoints)
    return np.sin(n_freq * x)

def gen_u_initial(n_xpoints,n_freq=4):
    """
    calculates u_initial = a0 + sum( a[n] cos nx + b[n] sin nx )
    """
    # generate IC
    a_0 = np.round((np.random.random() - 0.5) * 2, 2)
    u_initial = np.zeros(n_xpoints,) + a_0

    # Generate a and b arrays in one step
    a = np.round((np.random.random((n_freq,)) - 0.5) * 2, 2)
    b = np.round((np.random.random((n_freq,)) - 0.5) * 2, 2)

    # Calculate u_initial_cos and u_initial_sin 
    u_initial_cos_array = np.array([u_initial_cos(n_freq=i+1, n_xpoints=n_xpoints) for i in range(n_freq)]) #cos nx
    u_initial_sin_array = np.array([u_initial_sin(n_freq=i+1, n_xpoints=n_xpoints) for i in range(n_freq)]) #sin nx

    # Perform vectorized operations to calculate u_initial = a0 + sum( a[n] cos nx + b[n] sin nx )
    u_initial += np.dot(a, u_initial_cos_array) + np.dot(b, u_initial_sin_array)

    return u_initial

def simulate_IC(n_simulations,initial_conditions,L, n, t_max, dt, nu,
                plotting=False,time_res=None,space_res=None,keep_first_t=False,old_ic = False):
    if space_res is None:
        space_res = n
    if time_res is None:
        time_res = int(t_max/dt)
    if space_res > n or time_res > int(t_max/dt):
        raise ValueError("Invalid time_res or space_res")

    progress_bar = tqdm(total=n_simulations, desc="Simulations")
    x_grid = np.linspace(0, L, n)  # Spatial grid
    # Do it once to get the shapes (lazy way)
    x_grid,t_points,u_initial,u_solution = burgers_equation_simulation2(u_initial_1(L,n), x_grid, dt, t_max, nu,space_res,time_res,keep_first_t=False)
    
    u_t_train = np.zeros((n_simulations,u_solution.shape[1],u_solution.shape[0]))
    u_0_train = np.zeros((n_simulations,u_solution.shape[1],u_solution.shape[0]))
    i = 0
    while i < n_simulations:
        try:
            x_grid = np.linspace(0, L, n)  # Spatial grid
            
            if old_ic:
                w = np.round((np.random.random(8)) , 2)
                u_initial = np.zeros_like(x_grid)
                for idx, (name, init_func) in enumerate(initial_conditions.items()):
                    u_initial += init_func(L, n) * w[idx]
            else:
                u_initial = gen_u_initial(n,n_freq=4)

            x_grid,t_points,u_initial,u_solution = burgers_equation_simulation2(u_initial, x_grid, dt, t_max, nu,space_res,time_res,keep_first_t=False)

            if np.isnan(u_solution).any(): raise ValueError('NaN values in array')
            
            # Plotting
            if plotting:
                plt.figure(figsize=(10, 6))
                plt.plot(x_grid,u_initial, label='Initial Condition')
                for j in range(0, len(t_points), int(len(t_points) / 10)):  
                    plt.plot(x_grid, u_solution[:, j], label=f"t={t_points[j]:.2f}")
                plt.title("Solution of Burger's Equation")
                plt.xlabel("x")
                plt.ylabel("u(x, t)")
                plt.grid(True)
                plt.legend()
                plt.show()  
            u_t_train[i] = (u_solution.T)[None,:]
            u_0_train[i] = np.tile(u_initial, (len(t_points), 1))
            
            # Update progress bar
            progress_bar.update(1)
            i += 1
        except ValueError as e:
            print(f"Error in simulation {i + 1}: {e}. Simulation skipped")
            # Handle the error here, for example:
            # continue  # Skip this simulation and move to the next one
            pass

    progress_bar.close()
    return x_grid,u_t_train, u_0_train


def plot_ICs(L, n):

    # Define the periodic domain [0, L]
    x = np.linspace(0, L, n)

    # Plot the initial conditions
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 2, 1)
    plt.plot(x, u_initial_1(L, n), label='u_initial_1')
    plt.legend()

    plt.subplot(5, 2, 2)
    plt.plot(x, u_initial_2(L, n), label='u_initial_2')
    plt.legend()

    plt.subplot(5, 2, 3)
    plt.plot(x, u_initial_3(L, n), label='u_initial_3')
    plt.legend()

    plt.subplot(5, 2, 4)
    plt.plot(x, u_initial_4(L, n), label='u_initial_4')
    plt.legend()

    plt.subplot(5, 2, 5)
    plt.plot(x, u_initial_5(L, n), label='u_initial_5')
    plt.legend()

    plt.subplot(5, 2, 6)
    plt.plot(x, u_initial_6(L, n), label='u_initial_6')
    plt.legend()

    plt.subplot(5, 2, 7)
    plt.plot(x, u_initial_7(L, n), label='u_initial_7')
    plt.legend()

    plt.subplot(5, 2, 8)
    plt.plot(x, u_initial_8(L, n), label='u_initial_8')
    plt.legend()

    plt.tight_layout()
    plt.show()

def load_data(directory):
    metadata_path = os.path.join(directory, 'metadata.npy')
    u_0_train_path = os.path.join(directory, 'u_0_train.npy')
    u_t_train_path = os.path.join(directory, 'u_t_train.npy')
    x_grid_path = os.path.join(directory, 'x_grid.npy')

    metadata = np.load(metadata_path, allow_pickle=True)
    u_0_train = np.load(u_0_train_path, allow_pickle=True)
    u_t_train = np.load(u_t_train_path, allow_pickle=True)
    x_grid = np.load(x_grid_path, allow_pickle=True,)

    return metadata, u_0_train, u_t_train, x_grid


def prepare_data(u_0_train,u_t_train,device,t_max,L,batch_size=32):

    # Downsample the training data
    X = torch.tensor(u_0_train, dtype=torch.float32).to(device)
    y = torch.tensor(u_t_train, dtype=torch.float32).to(device)
    X = X.unsqueeze(1)
    y = y.unsqueeze(1)

    train_db = TensorDataset(X,y)
    
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False if device == 'cuda' else True,
        persistent_workers=False,
    )

    reduce_dims = list(range(X.ndim))
    #input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
    #input_encoder.fit(X)
    pos_encoding = PositionalEmbedding2D(grid_boundaries=[[0,t_max],[0,L]])


    # if encoding == "channel-wise":
    #     reduce_dims = list(range(y_train.ndim))
    # elif encoding == "pixel-wise":
    #     reduce_dims = [0]

    # output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
    # output_encoder.fit(y_train)

    data_processor = DefaultDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        positional_encoding=pos_encoding
    )
    data_processor = data_processor.to(device)
    return train_loader,data_processor

def generate_test_tensor(n_simulations,test_resolutions,device,L,n,t_max,dt,nu):
    # Test data
    # resolutions space_res x time_res
    initial_conditions = {
    'u_initial_1': u_initial_1,
    'u_initial_2': u_initial_2,
    'u_initial_3': u_initial_3,
    'u_initial_4': u_initial_4,
    'u_initial_5': u_initial_5,
    'u_initial_6': u_initial_6,
    'u_initial_7': u_initial_7,
    'u_initial_8': u_initial_8,
    }

    test_loaders = {}
    for res in test_resolutions:
        x_grid_test,u_t_test, u_0_test = simulate_IC(n_simulations,initial_conditions,L, n, t_max, dt, nu,
                                                    plotting=False,keep_first_t=False,
                                                    space_res=res[0],time_res=res[1])
        X_test = torch.tensor(u_0_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(u_t_test, dtype=torch.float32).to(device)
        X_test = X_test.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        test_db = TensorDataset(X_test,y_test)

        test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False if device == 'cuda' else True,
            persistent_workers=False,
        )
        
        test_loaders[f'{res[0]}x{res[1]}'] = test_loader
    return test_loaders