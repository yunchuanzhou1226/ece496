import time

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import pickle
import sys

sys.path.append('..')


num_steps = 6000
file_name = 'memory_2024-03-19-02_reward=-0.24_step=6000'
#rews_buf_GraphLDO_2023-11-22_noise=uniform_reward=0.00_ActorCriticGCN_rew_eng=True.npy
with open(SCH_PATH.joinpath(f'./saved_memories/{file_name}.pkl'), 'rb') as memory_file:
    memory = pickle.load(memory_file)

actions = memory.acts_buf[:num_steps]
print(actions)