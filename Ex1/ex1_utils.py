"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import copy
import math
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy

import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                         [0.59590059, -0.27455667, -0.32134392],
                         [0.21153661, -0.52273617, 0.31119955]]) # matrix for transform image to YIQ


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 205785926


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image=cv2.imread(filename)
    if representation == LOAD_RGB: #check if the representation.
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    elif representation == LOAD_GRAY_SCALE:
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invaild representation, can only be 1 or 2")
    image = image.astype(np.float_) / 255.0 # return the image with pixel value in a range of (0,1)
    return image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    if representation == LOAD_RGB:
        img = imReadAndConvert(filename, LOAD_RGB) # convert the image by the representation
        plt.imshow(img)
    else:
        img = imReadAndConvert(filename, LOAD_GRAY_SCALE)
        plt.imshow(img, cmap='gray') # case of Gray image
    plt.show()




    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    """
    take the ndarray of the image and reshape the image to a 3 columns that every column represent a channel,
    every channel is multiplied by the value of the yiq_from_rgb.transpose():
    R  G  B
    |  |  |
    |  |  |  *  yiq_from_rgb.transpose()
    |  |  |
    and the we reshape the image to the original shape of a ndarry of a image
    """
    OrignalShape = imgRGB.shape # save the shape of the picture
    return np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrignalShape)
    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    """
    Y  I  Q
    |  |  |
    |  |  |  *  yiq_from_rgb^(-1).transpose()
    |  |  |
    """
    OrigShape=imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1,3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)

    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    bins = np.zeros((256))
    normalizedImg = (255*(imgOrig - np.min(imgOrig))/np.ptp(imgOrig)).astype(int)
    normalizedImg = cv2.normalize(imgOrig, normalizedImg, 0, 255, cv2.NORM_MINMAX).astype(int) #normalize the image to a range of integer (0,255)


    for i in range(normalizedImg.shape[1]):  #Calculate bins of pixels between 0 - 255
        for j in range(normalizedImg.shape[0]):
            n = normalizedImg[j][i]
            bins[n] = bins[n] + 1

    cumm_sum = 0
    CFD = []

    for i in range(len(bins)): # calculate the cummsum
        cumm_sum = cumm_sum + bins[i]
        CFD.append(cumm_sum)

    CFD = CFD / cumm_sum # normalize the cumsum

    img2 = np.copy(normalizedImg)

    for i in range(img2.shape[1]):
        for j in range(img2.shape[0]):
            n = img2[j][i]
            img2[j][i] = CFD[n] * 255 #put rhe new normalize cumsum in a new image
    histImg = calHist(normalizedImg)
    histImgEq = calHist(img2)
    return img2, histImg, histImgEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    imgCopy = imOrig.copy()
    numDim = imOrig.ndim
    if numDim == 3: # check if the image is RGB or gray
        w, h, _ = imOrig.shape
        imgYIQ = transformRGB2YIQ(imOrig)
        imgCopy = imgYIQ[:, :, 0] # use only Y channel in the YIQ
    else:
        print("check if gray")
        w, h = imOrig.shape
    hist, bins = np.histogram(imgCopy * 255, bins=range(0, 256)) #get the histogram and bins
    Q = np.zeros(nQuant) #means
    Z = np.zeros(nQuant + 1).astype("uint") #divide to nQuant colors
    Z[0] = 0
    Z[-1] = 255
    pixelsInEach = w*h/nQuant
    sum_ = 0
    index = 1
    imgCopy = imgCopy * 255
    newImage = imgCopy.copy()
    errors = []
    images = []
    for i in range(0, 255):  # border init
        sum_ += hist[i]
        if sum_ >= pixelsInEach:
            Z[index] = i
            index += 1
            sum_ = 0
            if index == nQuant:
                break

    for j in range(0, nIter):
        for i in range(1, nQuant + 1):  # calc means
            start = Z[i - 1]
            end = Z[i]
            mean = np.sum(hist[start:end]*bins[start:end]) / np.sum(hist[start:end])
            Q[i - 1] = mean

        for i in range(1, nQuant):  # calc new borders (correct mistakes)
            newJump = (Q[i - 1] + Q[i]) / 2
            newJump = round(newJump)
            Z[i] = newJump

        index = nQuant - 1
        while index >= 0:  # apply changes
            start = Z[index + 1]
            newVal = Q[index]
            newImage[imgCopy <= start] = newVal
            index -= 1
        newImage = newImage / 255
        imageToAppend = newImage.copy()
        if numDim == 3:
            imgYIQ[:, :, 0] = newImage
            imageToAppend = transformYIQ2RGB(imgYIQ)
        images.append(imageToAppend)
        mse = np.mean(np.power(imOrig*255 - imageToAppend*255, 2)) #calculate errors
        errors.append(mse)

    return images, errors


def calHist(img: np.ndarray) -> np.ndarray: # calculate histogram function
    img_flat = img.ravel()
    hist = np.zeros(256)

    for pix in img_flat:
        hist[pix] += 1

    return hist

