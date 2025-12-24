import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from osmnx import projection as ox_projection

from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score

import Network

if __name__ == "main":
    ne = Network.NetworkExtractor("Jhapa, Nepal")
