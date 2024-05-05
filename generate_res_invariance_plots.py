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


