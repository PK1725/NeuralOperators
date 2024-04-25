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
import os
import uuid

data_path = "data/data1000_80_50/"

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
print("(available memory,total memory): ", torch.cuda.mem_get_info())

train_loader,data_processor = utils.prepare_data(u_0_train,u_t_train,device,t_max,L)

# generate test data
test_resolutions = [[space_res,time_res], [25,25],[50,50],[100,100]]
n_simulations = 5 # Number of simulations per resolution
test_loaders = utils.generate_test_tensor(n_simulations,test_resolutions,device,L,n,t_max,dt,nu)
print("Generated test data")

model = TFNO(n_modes=(16,16)
             , hidden_channels=32, in_channels=3,projection_channels=64, factorization='tucker', rank=0.42)
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

train_loss = h1loss
eval_losses={'l2': l2loss} # 'h1': h1loss

callback = callbacks.BasicLoggerCallback()

trainer = Trainer(model=model, n_epochs=200,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  callbacks=[callback],
                  verbose=True)

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders, # Note: can be set to dict() if no test set is available
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

# Generate a unique ID for the folder name
folder_id = str(uuid.uuid4())

# Create the folder
folder_path = os.path.join("models", f"model_{folder_id}")
os.makedirs(folder_path, exist_ok=True)

# Save the model state dictionary
model_path = os.path.join(folder_path, "model_checkpoint.pth")
torch.save(model.state_dict(), model_path)

# Save the training metadata
metadata_path = os.path.join(folder_path, "metadata.txt")
with open(metadata_path, "w") as f:
    f.write(str(metadata))

print("Model and metadata saved in folder:", folder_path)
