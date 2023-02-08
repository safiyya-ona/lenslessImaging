import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv
import torch


def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)


def lerp(a, b, x):
    """Linear interpolation"""
    return a + x * (b - a)


def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h, x, y):
    """Converts h to the right gradient vector and returns the dot product with (x,y)"""
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y


def normalize(arr):
    # Calculate the minimum and maximum values in the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Subtract the minimum value from all elements in the array
    arr = arr - min_val

    # Divide all elements in the array by the range of values
    arr = arr / (max_val - min_val)

    return arr


def sample_perlin_psf(x, y):
    xlin = np.linspace(0, x // 20, x, endpoint=False)
    ylin = np.linspace(0, y // 20, y, endpoint=False)
    xs, ys = np.meshgrid(xlin, ylin)
    psf = perlin(xs, ys, seed=random.randint(4, 100))
    psf = (normalize(psf) * 255).astype(np.uint8)
    return psf


def sample_psf(x, y):
    perlin_psf = sample_perlin_psf(x, y)
    psf = cv.Canny(perlin_psf, 0, 100)
    return psf


def sample_psf_as_tensor(x, y):
    psf = sample_psf(x, y)
    psf = psf.reshape((1, x, y))
    psf = torch.from_numpy(psf).float()
    return psf


if __name__ == "__main__":
    psf = sample_psf(256, 256)
    plt.imshow(psf,
               origin='upper', cmap='gray')

    plt.show()

    psf_tensor = sample_psf_as_tensor(256, 256)
