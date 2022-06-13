import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature

import skimage.io
from skimage.color import rgb2gray



from time import sleep
from timer_py import Timer


from matplotlib import pyplot as plt
import matplotlib.image as mpimg

t = Timer()
t.start()


img = mpimg.imread('C:/Users/ayoub/Desktop/CannyFilter/balle.png')



R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
image = 0.2989 * R + 0.5870 * G + 0.1140 * B
# plt.imshow(image, cmap='gray')
# plt.show








# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image)
# edges2 = feature.canny(image, sigma=3)

# display results
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
#
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('noisy image', fontsize=20)
#
# ax[1].imshow(edges1, cmap='gray')
# ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)
#
# ax[2].imshow(edges2, cmap='gray')
# ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

# for a in ax:
#     a.axis('off')
#
# fig.tight_layout()
# plt.show()

t.stop()
