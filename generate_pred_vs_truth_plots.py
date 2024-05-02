import torch
import utils
import os
import torch
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor
from neuralop.models import TFNO

from matplotlib import pyplot as plt
import numpy as np
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# Specify the directory
directory = 'models/new_IC_1000_80_50/'

# Load the model
model_path = os.path.join(directory, 'model_full.pth')
model = torch.load(model_path,map_location=device)

# model = TFNO()
# model.load_state_dict(state_dict)

# Load the metadata
metadata_path = os.path.join(directory, 'metadata.txt')
with open(metadata_path, 'r') as file:
    metadata = eval(file.read())

# Extract the variables from the metadata
L = metadata['L']
t_max = metadata['t_max']
nu = metadata['nu']
n = metadata['n']
dt = metadata['dt']
space_res = metadata['space_res']
time_res = metadata['time_res']


# Test data
# resolutions space_res x time_res
test_resolutions = [[space_res,time_res], [25,25],[50,50],[100,100]]
n_simulations = 1 # Number of simulations per resolution
test_loaders = {}
for res in test_resolutions:
    np.random.seed(666)
    x_grid_test,u_t_test, u_0_test = utils.simulate_IC(n_simulations,initial_conditions,L, n, t_max, dt, nu,
                                                plotting=False,keep_first_t=False,
                                                space_res=res[0],time_res=res[1],old_ic=False)
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
        pin_memory=True,
        persistent_workers=False,
    )
    test_loaders[f'{res[0]}x{res[1]}'] = test_loader


pos_encoding = PositionalEmbedding2D(grid_boundaries=[[0,t_max],[0,L]])
data_processor = DefaultDataProcessor(
    in_normalizer=None,
    out_normalizer=None,
    positional_encoding=pos_encoding
)

m = len(test_loaders)
fig, axs = plt.subplots(3, m, figsize=(16, 10.5))

for i, (resolution, loader) in enumerate(test_loaders.items()):
    sample = loader.dataset[0]
    space_res, time_res = resolution.split('x')
    space_res = int(space_res)
    time_res = int(time_res)
    x_grid = np.linspace(0, L, n)  # Spatial grid
    u_0 = sample['x'].detach().cpu().numpy()
    u_t = sample['y'].detach().cpu().numpy()
    #u_t_pred = model(data_processor(sample['x'])).detach().cpu().numpy()
    u_initial = u_0[0][0]

    # Plot the data in the corresponding subplot
    axs[0, i].imshow(u_t.squeeze(), aspect='auto',extent=[0,L,t_max,0], cmap='jet')

    axs[0, i].set_xlabel('x')
    axs[0, i].set_ylabel('t')
    data_processor.preprocess(sample, batched=False)['x']
    axs[1, i].imshow(model(sample['x'].unsqueeze(0)).squeeze().detach().cpu().numpy(), aspect='auto',extent=[0,L,t_max,0], cmap='jet')
    axs[1, i].set_title(f'Prediction')
    axs[1, i].set_xlabel('x')
    axs[1, i].set_ylabel('t')
    
    diff = abs(u_t.squeeze()-model(sample['x'].unsqueeze(0)).squeeze().detach().cpu().numpy())
    loss = round(np.mean(diff**2).item(),5)
    print(loss)
    if i == 0:
        axs[0, i].set_title(f'Training resolution: {resolution}, \n L2 error {loss} \n\n Ground Truth')
    else:
        axs[0, i].set_title(f'Resolution: {resolution}, \n L2 error {loss} \n\n Ground Truth')
    im = axs[2, i].imshow(diff, aspect='auto',extent=[0,L,t_max,0],cmap='hot')
    axs[2, i].set_title(f'Abs difference')
    axs[2, i].set_xlabel('x')
    axs[2, i].set_ylabel('t')
    # Colorbar for differences
    cbar = fig.colorbar(im, ax=axs[2, i], orientation='horizontal', pad=0.2)
    cbar.set_label('Abs Difference values')
    cbar.set_ticks(np.linspace(0,np.max(diff),4,endpoint=True))  # Add tick for largest possible value
fig.suptitle(f"Predictions on test data for {m} different resolutions and {m} different initial conditions\n(Burgers' Equation)", fontsize=18)

plt.tight_layout()
#plt.show()


# Save the figure
os.makedirs(f'figs/{directory}', exist_ok=True)
fig.savefig(f"figs/{directory}pred_vs_truth.png")

