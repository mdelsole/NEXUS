

import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import numpy as np
import skimage.transform, skimage.util


def calc_degree_of_gaussians(img):

    img = mpimg.imread(img)
    # extract grey values
    img = img[:,:,0]

    # Receptive Field function

    def gaussian2D(x, y, sigma):
        return (1.0/(1*math.pi*(sigma**2)))*math.exp(-(1.0/(2*(sigma**2)))*(x**2 + y**2))

    # Since scipy's convolve function doesn't accept functions, we need to sample the function

    # Make a matrix from the function
    def receptiveFieldMatrix(func):
        h = 30
        g = np.zeros((h,h))
        for xi in range(0,h):
            for yi in range(0,h):
                x = xi-h/2
                y = yi-h/2
                g[xi, yi] = func(x,y)
        return g

    def plotFilter(fun):
        g = receptiveFieldMatrix(fun)
        plt.imshow(g, cmap=plt.get_cmap('gray'))

    # The mexican hat function is a difference of gaussians, which leads to an on-center, off-surround receptive field,
    # found in retinal cells and vision neurons. It becomes a basic edge detector


    def mexicanHat(x,y,sigma1,sigma2):
        return gaussian2D(x,y,sigma1) - gaussian2D(x,y,sigma2)


    # Create the filter that resembles the retina
    plotFilter(lambda x,y: mexicanHat(x,y,3,4))

    # And now we use our filter that resembles the retina and run it over the image.
    # Convolution means applying the filter to the input. When applying the gaussian filter, every neuron in the output
    # layer is excited by nearby image neurons. The result of the convolution can then also be visualized in an image

    img_hat = signal.convolve(img,receptiveFieldMatrix(lambda x,y: mexicanHat(x,y,2,3)), mode='same')
    img_hat = skimage.transform.resize(img_hat, (12 , 12))

    return img_hat

