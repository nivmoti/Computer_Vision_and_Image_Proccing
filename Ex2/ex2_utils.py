import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import cv2


from matplotlib import pyplot as plt

def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 205785926

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    kernel = k_size[::-1]
    list= [
        np.dot(
            in_signal[max(0,i):min(i+len(kernel),len(in_signal))],
            kernel[max(-i,0):len(in_signal)-i*(len(in_signal)-len(kernel)<i)],
        )
        for i in range(1-len(kernel),len(in_signal))
    ]
    output=np.array(list)
    return output

def conv_flip(kernel: np.ndarray):
    kernelCopy=kernel.copy()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            kernelCopy[i][j]=kernel[kernel.shape[0]-i-1][kernel.shape[1]-j-1]
    return kernelCopy



def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    #Flip the kernel
    # kernelNew = conv_flip(kernel)
    # imageh=in_image.shape[0]
    # imagew=in_image.shape[1]
    # kernel_h=kernelNew.shape[0]
    # kernel_W = kernelNew.shape[1]
    # h=kernel_h//2
    # w = kernel_W // 2
    # imageNew=np.zeros(in_image.shape)
    # for i in range(h,imageh-h):
    #     for j in range(w,imagew-w):
    #         sum=0
    #         for m in range(kernel_h):
    #             for n in range(kernel_W):
    #                 sum=sum+kernelNew[m][n]*in_image[i-h+m][j-w+n]
    #         imageNew[m][n]=sum


    #return imageNew
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(in_image)

    # Add zero padding to the input image
    image_padded = np.zeros((in_image.shape[0] + 2, in_image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = in_image

    # Loop over every pixel of the image
    for x in range(in_image.shape[1]-kernel.shape[1]):
        for y in range(in_image.shape[0]-kernel.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y: y+kernel.shape[0], x: x+kernel.shape[1]]).sum()

    return output



def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernal=np.array([[1, 0, -1]])
    X = conv2D(in_image,kernal)
    kernalY=kernal.T
    Y = conv2D(in_image,kernalY)
    ori = np.arctan2(Y, X).astype(np.float64)
    mag = np.sqrt(X ** 2 + Y ** 2).astype(np.float64)

    return (ori,mag)


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaus_1d = getgaussian(k_size)
    return conv2D(in_image, np.dot(gaus_1d, np.transpose(gaus_1d)))


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    return cv2.GaussianBlur(in_image, (k_size, k_size), 0)
def gaussigma(k_size: int):
    return 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8


def getgaussian(size: int) -> np.ndarray:
    gas = np.array([[np.exp(-(np.square(x - (size - 1) / 2)) / (2 * np.square(gaussigma(size))))
                     for x in range(size)]])
    gas /= gas.sum()
    return np.transpose(gas)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    blur = cv2.GaussianBlur(img,(3,3),0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    img_crossing = laplacian / laplacian.max()
    zero_crossing_img = np.zeros(laplacian.shape)
    neg_pixel_count = 0
    pos_pixel_count = 0
    img_height, img_width = laplacian.shape
    # Looking for zero crossing patterns: such as {+,0,-} or {+,-}
    for i in range(1, img_height - 1):
        for j in range(1, img_width - 1):
            # 3x3 kernel
            pixel_neighbours = [img_crossing[i + 1, j - 1], img_crossing[i + 1, j],
                                img_crossing[i + 1, j + 1], img_crossing[i, j - 1],
                                img_crossing[i, j + 1],     img_crossing[i - 1, j - 1],
                                img_crossing[i - 1, j],     img_crossing[i - 1, j + 1]]

            for pixel_value in pixel_neighbours:
                if isPositive(pixel_value):
                    pos_pixel_count += 1
                elif not isPositive(pixel_value):
                    neg_pixel_count += 1

            # Checking if both the positive and negative value counts are positive,
            # then zero crossing potentially exists for that pixel
            zero_crossing = isPositive(pos_pixel_count) and isPositive(neg_pixel_count)

            # Finding the maximum neighbour pixel difference and changing the pixel value
            min_value_diff = img_crossing[i, j] + np.abs(min(pixel_neighbours))
            max_value_diff = np.abs(img_crossing[i, j]) + max(pixel_neighbours)
            if zero_crossing:
                if isPositive(img_crossing[i, j]):
                    zero_crossing_img[i, j] = min_value_diff
                elif not isPositive(img_crossing[i, j]):
                    zero_crossing_img[i, j] = max_value_diff

    return zero_crossing_img

def isPositive(value):
    return value > 0

def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    edge_image = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(image=edge_image, threshold1=50, threshold2=200)  # Canny Edge Detection
    cv2.imshow('Edge Image', edges)
    circles = FindHoughCircles(edges, min_radius, max_radius,0.5)


    return circles


def FindHoughCircles(edge_image, r_min, r_max, threshold):
    # image size
    img_height, img_width = edge_image.shape


    ## Thetas is bins created from 0 to 360 degree with increment of 3 to run faster
    thetas = np.arange(0, 360, step=2)

    ## Radius ranges from r_min to r_max with 2 step to run faster
    rs = np.arange(r_min, r_max, step=5)

    # Calculate Cos(theta) and Sin(theta)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    circle_candidates = []
    for r in rs:
        for t in range(len(thetas)):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
    accumulator = defaultdict(int)

    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0:  # white pixel
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1  # vote for current candidate
    out_circles = []
    for candidate_circle, votes in accumulator.items():
        x, y, r = candidate_circle
        current_vote_percentage = votes / len(thetas)
        if current_vote_percentage >= threshold: #do remove "unwanted" circles
            # Shortlist the circle for final result
            out_circles.append((x, y, r))
    return out_circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    filtered_image_OpenCV = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    filter=bilateral_filter(in_image,sigma_space,sigma_color,k_size)


    return (filtered_image_OpenCV,filter)

def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)


def bilateral_filter(image, sigma_space, sigma_color,k_size):

    kernel_size = k_size
    half_kernel_size = int(kernel_size / 2)
    result = np.zeros(image.shape)
    W = 0

    # Iterating over the kernel
    for x in range(-half_kernel_size, half_kernel_size+1):
        for y in range(-half_kernel_size, half_kernel_size+1):
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            shifted_image = np.roll(image, [x, y], [1, 0])
            intensity_difference_image = image - shifted_image
            Gcolor = gaussian(
                intensity_difference_image ** 2, sigma_color)
            result += Gspace*Gcolor*shifted_image
            W += Gspace*Gcolor

    return result / W

