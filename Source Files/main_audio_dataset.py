#Necessary Imports

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances
from scipy.signal import stft
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchaudio
from itertools import combinations
import h5py
import warnings
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
from collections import defaultdict
import cv2

warnings.filterwarnings("ignore")

# Mount Google Drive (if needed for data)
drive.mount('/content/drive')
