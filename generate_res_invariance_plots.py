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

test_resolutions = [[space_res,time_res], [2**4,2**4],[2**5,2**5],[2**6,2**6],[2**7,2**7],[2**8,2**8],[2**9,2**9],[2**10,2**10]]

# Specify the directory
directory = 'models/04-05/best_checkpoint'
# Load the model
model_path = os.path.join(directory, 'best_model_state_dict.pt')
model = torch.load(model_path,map_location=device)


for res in test_resolutions:
    data_path = f"data/test_data/data100_{res[0]}_{res[1]}"
    metadata, u_0_train, u_t_train, x_grid = utils.load_data(data_path)

    train_loader,data_processor = utils.prepare_data(u_0_train,u_t_train,device,t_max,L,batch_size=100)
    for sample in train_loader:
        x = sample['x']
        print(x.shape)
        break
    print(train_loader.dataset['x'].shape)
    data_processor = data_processor.to(device)