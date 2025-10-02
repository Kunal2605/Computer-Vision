import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
img = cv2.imread('lena_gray_512.tif', cv2.IMREAD_GRAYSCALE)

# Negative transformation
negative = 255 - img

# Logarithmic transformation
c_log = 1
log_trans = c_log * np.log(1 + img.astype(np.float32))
log_trans = np.array(log_trans, dtype=np.uint8)

# Gamma transformation (gamma = 2.2)
gamma = 2.2
c_gamma = 255 / (np.max(img) ** gamma)
gamma_trans = c_gamma * (img.astype(np.float32) ** gamma)
gamma_trans = np.array(gamma_trans, dtype=np.uint8)

# Piecewise linear transformation
r1, s1 = 70, 100
r2, s2 = 140, 180
piecewise = np.zeros_like(img)

# Apply piecewise linear mapping
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r = img[i, j]
        if r < r1:
            piecewise[i, j] = s1
        elif r < r2:
            piecewise[i, j] = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
        else:
            piecewise[i, j] = s2
piecewise = piecewise.astype(np.uint8)

# Plotting all transformations
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
axes[0].set_title('Original')
axes[0].imshow(img, cmap='gray')
axes[0].axis('off')

axes[1].set_title('Negative')
axes[1].imshow(negative, cmap='gray')
axes[1].axis('off')

axes[2].set_title('Logarithmic')
axes[2].imshow(log_trans, cmap='gray')
axes[2].axis('off')

axes[3].set_title('Gamma (2.2)')
axes[3].imshow(gamma_trans, cmap='gray')
axes[3].axis('off')

axes[4].set_title('Piecewise Linear')
axes[4].imshow(piecewise, cmap='gray')
axes[4].axis('off')

plt.tight_layout()
plt.show()