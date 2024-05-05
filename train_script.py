import numpy as np
import utils,sys
import torch
from neuralop.models import TFNO
from neuralop import Trainer
import neuralop.training.callbacks as callbacks
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor
import wandb
import os
import uuid

data_path = "data/data1000_128_64"

metadata, u_0_train, u_t_train, x_grid = utils.load_data(data_path)
metadata = metadata.item()

print("Loaded data shapes:")
print(u_0_train.shape,u_t_train.shape,x_grid.shape)
print(u_t_train.nbytes)

L = metadata['L']
t_max = metadata['t_max']
nu = metadata['nu']
n = metadata['n']
dt = metadata['dt']
space_res = metadata['space_res']
time_res = metadata['time_res']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device:", device)
if device == 'cuda': print("(available memory,total memory): ", torch.cuda.mem_get_info())

train_loader,data_processor = utils.prepare_data(u_0_train,u_t_train,device,t_max,L,batch_size=16)
print("Loaded train data")
# generate test data
test_resolutions = [[space_res,time_res], [2**4,2**4],[2**5,2**5],[2**9,2**9]]
#test_resolutions = [[space_res,time_res]]
n_simulations = 5 # Number of simulations per resolution
test_loaders = utils.generate_test_tensor(n_simulations,test_resolutions,device,L,n,t_max,dt,nu,old_ic=False,solver='Fourier')
print("Generated test data")

model = TFNO(n_modes=(16,16), hidden_channels=32, in_channels=3,projection_channels=64,n_layers=4,stabilizer='tanh')
#TFNO(n_modes=(16,16), hidden_channels=32, in_channels=3,projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()
optimizer = torch.optim.Adam(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = l2loss
eval_losses={'l2': l2loss} # 'h1': h1loss

# Generate a unique ID for the folder name
folder_id = str(uuid.uuid4())

# Create the folder
folder_path = os.path.join("models", f"model_{folder_id}")
os.makedirs(folder_path, exist_ok=True)

callback = callbacks.BasicLoggerCallback()
callbackCheckpoint_saver = utils.CheckpointCallbackAdjusted(save_dir=f"{folder_path}/best_checkpoint/",save_best='l2')

trainer = Trainer(model=model, n_epochs=300,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  callbacks=[callbackCheckpoint_saver,callback],
                  verbose=True)

data_processor = data_processor.to(device)

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders, # Note: can be set to dict() if no test set is available
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)
#load best model

# Specify the directory
load_dir = f"{folder_path}/best_checkpoint"

# Load the model
model_path = os.path.join(load_dir, 'best_model_state_dict.pt')
state_dict = torch.load(model_path,map_location=device)

model.load_state_dict(state_dict)

# Save the model state dictionary
model_path = os.path.join(folder_path, "model_checkpoint.pth")
torch.save(model.state_dict(), model_path)
# Save the model state dictionary
model_path = os.path.join(folder_path, "model_full.pth")
torch.save(model, model_path)

# Save the training metadata
metadata_path = os.path.join(folder_path, "metadata.txt")
with open(metadata_path, "w") as f:
    f.write(str(metadata))

print("Model and metadata saved in folder:", folder_path)


# Initialize grid
L = 2*np.pi  # Spatial domain length
T = 1.6037  # Time domain length
N = 2**11  # Number of grid points
x = np.linspace(0, L, N)
u0 = -np.sin(x-np.pi)  # Initial condition

x_grid,t_points,u_initial,u_solution = utils.burgers_equation_simulation2(u0, x, dt, T, nu,64,64,keep_first_t=False,solver='Fourier')


import matplotlib.pyplot as plt
def plot_shock(x_grid,t_points,u_initial,u_solution):
  u_t_train = np.zeros((1,u_solution.shape[1],u_solution.shape[0]))
  u_0_train = np.zeros((1,u_solution.shape[1],u_solution.shape[0]))
  u_t_train[0] = (u_solution.T)[None,:]
  u_0_train[0] = np.tile(u_initial, (len(t_points), 1))

  # Downsample the training data
  X = torch.tensor(u_0_train, dtype=torch.float32).to(device)
  y = torch.tensor(u_t_train, dtype=torch.float32).to(device)
  X = X.unsqueeze(1)
  y = y.unsqueeze(1)

  train_db = TensorDataset(X,y)

  train_loader = torch.utils.data.DataLoader(
      train_db,
      batch_size=1,
      shuffle=True,
      num_workers=0,
      pin_memory=False if device == 'cuda' else True,
      persistent_workers=False,
  )

  reduce_dims = list(range(X.ndim))
  #input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
  #input_encoder.fit(X)
  pos_encoding = PositionalEmbedding2D(grid_boundaries=[[0,T],[0,L]])


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
  sample = train_loader.dataset[0]
  u_0 = sample['x'].detach().cpu().numpy()
  u_t = sample['y'].detach().cpu().numpy()
  u_initial = u_0[0][0]
  data_processor.preprocess(sample, batched=False)['x']
  pred = model(sample['x'].unsqueeze(0)).squeeze().detach().cpu().numpy()
  # Plot results
  plt.figure(figsize=(6, 6))
  plt.plot(x_grid, u_initial, label='Initial condition')
  plt.plot(x_grid, pred[-1,:], label='Prediction')
  plt.plot(x_grid, u_solution[:,-1], label='True solution')

  plt.xlabel('x')
  plt.ylabel('u')
  plt.title("Burger's equation solution")
  plt.legend()
  # Specify the directory for saving the plot
  figs_folder = os.path.join("figs", "etc", folder_id)
  os.makedirs(figs_folder, exist_ok=True)

  # Save the plot
  plot_path = os.path.join(figs_folder, "shock_plot.png")
  plt.savefig(plot_path)

plot_shock(x_grid,t_points,u_initial,u_solution)