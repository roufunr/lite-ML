import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend

import numpy as np
import matplotlib.pyplot as plt

# Provided array
pixel_intensities = np.array([0.0, 0.0, 0.0, 12.0, 13.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 16.0, 9.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0, 16.0, 6.0, 0.0, 0.0, 0.0, 7.0, 15.0, 16.0, 16.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 16.0, 16.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0, 16.0, 16.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 16.0, 10.0, 0.0, 0.0]
)

# Reshape the array into a 2D array with dimensions 8x8
image = pixel_intensities.reshape(8, 8)

# Plot the image
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
