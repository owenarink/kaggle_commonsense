# use as: from dependencies.dependencies import *
import os
import csv
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import matplotlib
if os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.set_cmap('cividis')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

from src.processing import preprocess
from src.features.tfidf import get_tfidf
from src.models.mlp_tfidf import MLP
from src.models.cnn_text import TextResNetCNN
from src.models.transformer import TextTransformer 

try:
    from src.modules import LinearModule
except Exception:
    LinearModule = None

try:
    from src.eval import evaluate_model
except Exception:
    evaluate_model = None

CNN_TFIDF = None
try:
    from src.models.cnn_text import CNN_TFIDF
except Exception:
    try:
        from src.models.cnn_text import CNN_TFIDF
    except Exception:
        CNN_TFIDF = None
