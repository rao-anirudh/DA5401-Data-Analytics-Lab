# Course: DA5401
# Name: Anirudh Rao
# Roll No: BE21B004
# Assignment: 1

# Importing the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the data of (x,y) coordinates

df = pd.read_csv("be21b004.csv", header=None, names=["X", "Y"])

# Discretization

df["X"] = df["X"].apply(lambda x: round(x))
df["Y"] = df["Y"].apply(lambda y: round(y))

# Converting to Boolean matrix representation

orig_matrix = np.zeros((1000, 1000)).astype(int)
for i in range(len(df)):
    orig_matrix[df["X"][i]][df["Y"][i]] = 1

# Helper function to convert Boolean matrix to (x,y) coordinates
def bool_to_coordinates(bool_matrix):
    x = []
    y = []
    for i in range(len(bool_matrix)):
        for j in range(len(bool_matrix)):
            if bool_matrix[i][j] == 1:
                x.append(i)
                y.append(j)
    return x, y


# Performing the matrix operations

rotated_matrix = np.flip(orig_matrix.T, axis=1)
flipped_matrix = np.flip(orig_matrix, axis=1)

# Visualisation

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
matrices = [orig_matrix, rotated_matrix, flipped_matrix]
titles = ["Original", "Rotated", "Flipped"]
for i in range(3):
    x, y = bool_to_coordinates(matrices[i])
    ax[i].scatter(x, y, c="black")
    ax[i].axis("off")
    ax[i].set_title(titles[i])
plt.show()
