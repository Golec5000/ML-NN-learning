import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("wombat.jpeg", cv2.IMREAD_GRAYSCALE)

# plt.imshow(image, cmap="gray")
# plt.axis("off")
# plt.show()

print(image)

print(image.shape)

image_bgr = cv2.imread("wombat.jpeg", cv2.IMREAD_COLOR)

# plt.imshow(image_bgr, cmap="gray")
# plt.axis("off")
# plt.show()


image_rgb = cv2.imread("wombat.jpeg", cv2.COLOR_BGR2RGB)

# image_rgb = cv2.resize(image_rgb,(50,50))

plt.imshow(image_rgb)
plt.axis("off")
# plt.show()

# kernel

# kernel used for blurring, sharpening, edge detection, and more.
# A kernel is a matrix, where the number of rows and columns is always odd.
# kernel example
# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
# It is a 3x3 kernel with 5 in the center and -1 at all other positions.
# This kernel is used for sharpening an image.

image = cv2.imread("wombat.jpeg", cv2.IMREAD_GRAYSCALE)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

sharped_image = cv2.filter2D(image, -1, kernel)

plt.imshow(sharped_image, cmap="gray")
plt.axis("off")
# plt.show()

image = cv2.imread("wombat.jpeg", cv2.IMREAD_GRAYSCALE)

image_enhanced = cv2.equalizeHist(image)  # histogram equalization

plt.imshow(image_enhanced, cmap="gray")
plt.axis("off")
# plt.show()

# Color isolation

image = cv2.imread("wombat.jpeg", cv2.IMREAD_COLOR)

immage_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV

lower_blue = np.array([50, 100, 50])
upper_blue = np.array([130, 255, 255])

mask = cv2.inRange(immage_hsv, lower_blue, upper_blue)

image_isolated = cv2.bitwise_and(image, image, mask=mask)

plt.imshow(image_isolated, cmap="gray")
plt.axis("off")
# plt.show()

# Thresholding

image = cv2.imread("wombat.jpeg", cv2.IMREAD_GRAYSCALE)

max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10

image_binarized = cv2.adaptiveThreshold(image, max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, neighborhood_size,
                                        subtract_from_mean)

plt.imshow(image_binarized, cmap="gray")
plt.axis("off")
# plt.show()


# Edge detection

image = cv2.imread("wombat.jpeg", cv2.IMREAD_GRAYSCALE)

median_intensity = np.median(image)

lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

image_canny = cv2.Canny(image, lower_threshold, upper_threshold)

plt.imshow(image_canny, cmap="gray")
plt.axis("off")
# plt.show()


# Creating features from images for machine learning

image = cv2.imread("wombat.jpeg", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("wombat.jpeg", cv2.IMREAD_COLOR)

image_resized = cv2.resize(image, (50, 50))  # resize the image to 100x100 pixels

image_flattened = image_resized.flatten()  # flatten the image to a 1D array

print(image_flattened)
print(image_flattened.shape)

plt.imshow(image_resized, cmap="gray")
plt.axis("off")
# plt.show()

# Using histograms as features

np.random.seed(0)
