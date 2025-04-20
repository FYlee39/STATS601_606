import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio  # To read JPEG images
import importlib.util
import os
from eRPCA_py import eRPCA
from Reproduction.func import *

# Load and normalize the reduced defect image
reduced_defect2 = imageio.imread("reduced_defect2.jpg").astype(np.float64).T

# Normalize the image
defect2_prob = reduced_defect2 / np.max(reduced_defect2)

# Initialize array to store samples
rows, cols = defect2_prob.shape
defect2_sample = np.empty((rows, cols, 500))

# Generate samples using binomial distribution
for i in range(rows):
    for j in range(cols):
        defect2_sample[i, j, :] = np.random.binomial(1, defect2_prob[i, j], 500)

# Plot the 'truth' image
plt.figure(figsize=(5, 5))
plt.imshow(defect2_prob, cmap='gray')
plt.axis('off')
plt.title("Truth")
plt.savefig("Truth.png")
plt.show()

# Plot the first 4 'observed' images in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for k in range(4):
    ax = axes[k // 2, k % 2]
    ax.imshow(defect2_sample[:, :, k], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Observed {k+1}")

plt.tight_layout()
plt.savefig("Observed.png")
plt.show()

erpca = eRPCA.ERPCA(observation_matrix=defect2_sample)
L_est, S_group_est = erpca.run()

# Enable LaTeX for titles
plt.rcParams['text.usetex'] = True

# Plot the estimated L matrix
plt.figure(figsize=(6, 6))
plt.imshow(L_est, cmap='gray')  # Python is 0-based
plt.axis('off')
plt.title(r'Estimated L ($e^{RPCA}$)', fontsize=20)
plt.savefig("erpca_L.png")
plt.show()

# Plot the estimated S matrix
plt.figure(figsize=(6, 6))
plt.imshow(S_group_est, cmap='gray')
plt.axis('off')
plt.title(r'Estimated S ($e^{RPCA}$)', fontsize=20)
plt.savefig("erpca_S.png")
plt.show()

# rpca
rpca = RPCA(observation_matrix=defect2_sample)
L_est, S_est = rpca.run()

# Plot the estimated L matrix
plt.figure(figsize=(6, 6))
plt.imshow(L_est, cmap='gray')  # Python is 0-based
plt.axis('off')
plt.title(r'Estimated L ($RPCA$)', fontsize=20)
plt.savefig("rpca_L.png")
plt.show()

# Plot the estimated S matrix
plt.figure(figsize=(6, 6))
plt.imshow(S_est, cmap='gray')
plt.axis('off')
plt.title(r'Estimated S ($RPCA$)', fontsize=20)
plt.savefig("rpca_S.png")
plt.show()

# epca
L_est = epca(defect2_sample, type="Bernoulli")

# Plot the estimated L matrix
plt.figure(figsize=(6, 6))
plt.imshow(L_est, cmap='gray')  # Python is 0-based
plt.axis('off')
plt.title(r'Estimated L ($ePCA$)', fontsize=20)
plt.savefig("epca_L.png")
plt.show()
