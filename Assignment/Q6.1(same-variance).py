###################~~ Problem-6 Part-1 ~~#####################
###################~~ binclass.txt analysis With Same Variance ~~#####################

import os
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

# Variables to be initialised before running the code
meshgrid_dim = 100                      # Specify the dimensions for the meshgrid matrix (meshgrid_dim * meshgrid_dim matrix)
filepath = "./binclass.txt"      # path of the dataset

if(os.path.exists(filepath)):
    data = pd.read_csv(filepath, header=None)
else:
    print(f"ERROR: The dataset file doesn't exist at {filepath}")
    sys.exit()

# Dataset divided into classes
pos = data[data[2] == 1]
neg = data[data[2] == -1]

# Means of the classes and the dataset
pos_mean = np.array([np.mean(pos[0]), np.mean(pos[1])]) 
neg_mean = np.array([np.mean(neg[0]), np.mean(neg[1])])
mean =  np.array([np.mean(data[0]), np.mean(data[1])]) 
print(f"Mean of dataset which belong to positive class {pos_mean}")
print(f"Mean of dataset which belong to negative class {neg_mean}")
print(f"Mean of dataset {mean}")

variance = 0.5* np.mean(np.array(np.linalg.norm(np.array(data[[0,1]])- mean, ord = 2, axis = 1))**2)
print(f"Variance of dataset {variance}")

# Creation of the meshgrid
x_min = data[0].min()-1
y_min = data[1].min()-1
x_max = data[0].max()+1
y_max = data[1].max()+1
h_x = (x_max-x_min)/meshgrid_dim
h_y = (y_max-y_min)/meshgrid_dim 
mg_x,mg_y = np.meshgrid(np.arange(x_min, x_max, h_x),np.arange(y_min, y_max, h_y))
(f_x,f_y) = mg_x.shape

def diff_prob(point):
    exp_pos = (np.array(np.linalg.norm(point- pos_mean))**2) / variance
    exp_neg = (np.array(np.linalg.norm(point- neg_mean))**2) / variance
    prob_pos = 0.5 * math.exp(-0.5*exp_pos)/variance
    prob_neg = 0.5 * math.exp(-0.5*exp_neg)/variance 
    return prob_pos-prob_neg

# matrix for storing the result of diff_prob for all the points of mesh_grid
f = np.zeros(mg_x.shape)
for i in range(0,f_x):
    for j in range(0,f_y):
        f[i][j] = diff_prob(np.array([mg_x[i][j], mg_y[i][j]]))

# plot the information
plt.contour(mg_x, mg_y, f,[0])
plt.scatter(pos[0],pos[1],color = 'r', cmap = 'Paired', label = "+1")
plt.scatter(neg[0],neg[1],color = 'b', cmap = 'Paired', label = "-1")
plt.legend()
plt.savefig("Q6.1(same-variance)_plot.png")
plt.show()