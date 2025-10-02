import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('jetplane.tif', cv2.IMREAD_GRAYSCALE)

# Prepare a list
bit_planes = []
for i in range(8):
    # Extract bit planes
    plane = np.bitwise_and(img, 1 << i) >> i
    bit_planes.append(plane)

# Reconstruct image
reconstructed = np.zeros_like(img, dtype=np.uint8)
for i in range(8):
    reconstructed += (bit_planes[i] << i)

# Plot original and 8 planes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Show original image
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Show bit planes
for i in range(8):
    row = 0 if i < 4 else 1
    col = i + 1 if i < 4 else i - 3
    axes[row, col].imshow(bit_planes[i], cmap='gray')
    axes[row, col].set_title(f'Bit Plane {i+1}')
    axes[row, col].axis('off')


axes[1, 0].imshow(reconstructed, cmap='gray')
axes[1, 0].set_title('Reconstructed')
axes[1, 0].axis('off')
plt.tight_layout()
plt.show()