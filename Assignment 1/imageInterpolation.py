import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('house.tif', cv2.IMREAD_GRAYSCALE)

# Downscale to 32x32
small = cv2.resize(img, (32, 32))

# Upscale using different interpolation methods
nearest = cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(small, (256, 256), interpolation=cv2.INTER_LINEAR)
cubic = cv2.resize(small, (256, 256), interpolation=cv2.INTER_CUBIC)


# Display results
plt.figure(figsize=(15,4))
plt.subplot(1,5,1)
plt.title('Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,5,2)
plt.title('Downscale to 32x32')
plt.imshow(small, cmap='gray')
plt.axis('off')

plt.subplot(1,5,3)
plt.title('Nearest Neighbor')
plt.imshow(nearest, cmap='gray')
plt.axis('off')

plt.subplot(1,5,4)
plt.title('Bilinear')
plt.imshow(linear, cmap='gray')
plt.axis('off')

plt.subplot(1,5,5)
plt.title('Bicubic')
plt.imshow(cubic, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()