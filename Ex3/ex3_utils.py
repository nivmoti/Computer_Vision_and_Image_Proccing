import math
import sys
from typing import List

import cv2
import numpy as np
import pygame as pygame
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 205785926


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    kernel = np.array([[-1, 0, 1]])
    X = cv2.filter2D(im2, -1, kernel)
    Y = cv2.filter2D(im2, -1, kernel.T)
    t_drive = im2 - im1
    points = []
    u_v = []
    for i in range(win_size, im1.shape[0] - win_size + 1, step_size):
        for j in range(win_size, im1.shape[1] - win_size + 1, step_size):
            xi = X[i - int(win_size / 2):i + int(win_size / 2), j - int(win_size / 2):j + int(win_size / 2)]
            yi = Y[i - int(win_size / 2):i + int(win_size / 2), j - int(win_size / 2):j + int(win_size / 2)]
            t = t_drive[i - int(win_size / 2):i + int(win_size / 2), j - int(win_size / 2):j + int(win_size / 2)]
            AtA = [[(xi * xi).sum(), (xi * yi).sum()],
                   [(xi * yi).sum(), (yi * yi).sum()]]
            lamdas = np.linalg.eigvals(AtA)
            lamda2 = np.min(lamdas)
            lamda1 = np.max(lamdas)
            if lamda2 <= 1 or lamda1 / lamda2 >= 100:
                continue
            arr = [[-(xi * t).sum()], [-(yi * t).sum()]]
            u_v.append(np.linalg.inv(AtA).dot(arr))
            points.append([j, i])
    return np.array(points), np.array(u_v)



def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    img1_py = gaussianPyr(img1, k)
    img2_py = gaussianPyr(img2, k)
    # entering the last pyramid
    points, u_v_prev = opticalFlow(img1_py[-1], img2_py[-1], stepSize, winSize)
    points = list(points)
    u_v_prev = [uv.flatten().tolist() for uv in u_v_prev]  # Convert arrays to lists
    for i in range(1, k):
        # find optical flow for this level
        point_i, uv_i = opticalFlow(img1_py[-1 - i], img2_py[-1 - i], stepSize, winSize)
        uv_i = [uv.flatten().tolist() for uv in uv_i]  # Convert arrays to lists
        point_i = list(point_i)
        for g in range(len(point_i)):
            point_i[g] = list(point_i[g])
        # update uv according to formula
        for j in range(len(points)):
            points[j] = [element * 2 for element in points[j]]
            u_v_prev[j] = [element * 2 for element in u_v_prev[j]]
        # If location of movements we found are new then append them, else add them to the proper location
        for j in range(len(point_i)):
            if point_i[j] in points:
                u_v_prev[j] += uv_i[j]
            else:
                points.append(point_i[j])
                u_v_prev.append(uv_i[j])
    # now we shall change uv and xy to a 3 dimensional array
    arr = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if [y, x] not in points:
                arr[x, y] = [0, 0]
            else:
                arr[x, y] = u_v_prev[points.index([y, x])]
    return arr

# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    T = np.eye(3)

    for _ in range(5):
        # Warp image using current translation matrix
        warped_im1 = cv2.warpPerspective(im1, T, (im1.shape[1], im1.shape[0]))

        # Compute error and gradient of warped image
        error = warped_im1 - im2
        k_mat=np.array([[-1, 1]])
        Ix_w = cv2.filter2D(warped_im1, -1, k_mat, borderType=cv2.BORDER_REPLICATE)
        Iy_w = cv2.filter2D(warped_im1, -1, k_mat.T, borderType=cv2.BORDER_REPLICATE)

        # Solve the system of equations
        A = np.array([[np.sum(Ix_w * Ix_w) + 0.01, np.sum(Ix_w * Iy_w)], [np.sum(Ix_w * Iy_w), np.sum(Iy_w * Iy_w) + 0.01]])
        b = np.array([-np.sum(Ix_w * error), -np.sum(Iy_w * error)])
        v = np.linalg.solve(A, b)

        # Limit maximum translation per iteration
        v[0] = np.abs(np.clip(v[0], -10, 10))
        v[1] = np.abs(np.clip(v[1], -10, 10))
        # Update translation matrix
        T[0, 2] -= v[0]
        T[1, 2] -= v[1]

    return T


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    x_1, y_1, x_2, y_2 = findXsYsCorr(im1, im2)
    return np.float32([[1, 0, x_2 - x_1 - 1], [0, 1, y_2 - y_1 - 1], [0, 0, 1]])
def findXsYsCorr(pic1, pic2):
    subtle_pading = np.max(pic1.shape) // 2
    pading1 = np.fft.fft2(np.pad(pic1, subtle_pading))
    pading2 = np.fft.fft2(np.pad(pic2, subtle_pading))
    prod = pading1 * pading2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + subtle_pading:-subtle_pading + 1, 1 + subtle_pading:-subtle_pading + 1]
    y1, x1 = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(pic2.shape) // 2
    return x1, y1, x2, y2


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass

def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    EPS = 0.000001
    min_error = np.inf
    final_rotation = np.eye(3, dtype=np.float32)
    directions = opticalFlow(im1, im2)[1]
    final_rotated_img = 0
    for u, v in directions:
        if u == 0:
            angle = 0
        else:
            angle = np.arctan(v / u)  # getting the angle
        check = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]], dtype=np.float_)
        rotated_img = cv2.warpPerspective(im1, check, im1.shape[::-1])  # rotating the image
        mse = np.square(im2 - rotated_img).mean()  # calculating the error relative to image 2
        if mse < min_error:
            min_error = mse
            final_rotation = check
            final_rotated_img = rotated_img.copy()
        if mse < EPS:
            break

    translation = findTranslationLK(final_rotated_img, im2)  # finding the translation from the rotated image to im2
    final_ans = translation @ final_rotation  # dot product for getting the rigid matrix
    return final_ans

def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    output = np.zeros_like(im2)

    # iterate over image 2
    for x in range(im2.shape[0]):
        for y in range(im2.shape[1]):
            # change the 2d pixel to 3d homagraphicaly
            pixel_3d = np.array([[x],
                                 [y],
                                 [1]])
            get_pixel_from_img1 = T @ pixel_3d
            img1_x = get_pixel_from_img1[0] / get_pixel_from_img1[2]
            img1_y = get_pixel_from_img1[1] / get_pixel_from_img1[2]

            # check if pixels are ints or floats
            float_x = img1_x % 1
            float_y = img1_y % 1

            # if they are float transform them from im2 according to formula
            if float_x != 0 or float_y != 0:
                output[x, y] = ((1 - float_x) * (1 - float_y) * im2[int(np.floor(img1_x)), int(np.floor(img1_y))]) \
                                + (float_x * (1 - float_y) * im2[int(np.ceil(img1_x)), int(np.floor(img1_y))]) \
                                + (float_x * float_y * im2[int(np.ceil(img1_x)), int(np.ceil(img1_y))]) \
                                + ((1 - float_x) * float_y * im2[int(np.floor(img1_x)), int(np.ceil(img1_y))])
            # if they are ints transform them as is
            else:
                img1_x = int(img1_x)
                img1_y = int(img1_y)
                output[x, y] = im2[img1_x, img1_y]
    return output


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    ans = []
    width = (2 ** levels) * int(img.shape[1] / (2 ** levels))
    height = (2 ** levels) * int(img.shape[0] / (2 ** levels))
    img = cv2.resize(img, (width, height))  # resize the image to dimensions that can be divided into 2 x times
    img = img.astype(np.float_)
    ans.append(img)  # level 0 - the original image
    temp_img = img.copy()
    for i in range(1, levels):
        temp_img = reduce(temp_img)  # 2 times smaller image
        ans.append(temp_img)
    return ans


def reduce(img: np.ndarray) -> np.ndarray:
    g_kernel = get_gaussian()
    blur_img = cv2.filter2D(img, -1, g_kernel)
    new_img = blur_img[::2, ::2]
    return new_img


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gauss_pyr = gaussianPyr(img, levels)
    g_kernel = get_gaussian() * 4
    for i in range(levels - 1):
        gauss_pyr[i] = gauss_pyr[i] - gaussExpand(gauss_pyr[i + 1], g_kernel)
    return gauss_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    temp = lap_pyr[-1]  # the smallest image (from the gaussPyramid)
    gs_k = get_gaussian() * 4
    i = levels - 1
    while i > 0:  # go through the list from end to start
        expand = gaussExpand(temp, gs_k)
        temp = expand + lap_pyr[i - 1]
        i -= 1
    return temp


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    img_1, img_2 = resizemask(img_1, img_2, mask)
    naive_blend = naive_blending(img_1, img_2, mask)
    lapla_a = laplaceianReduce(img_1, levels)
    lapla_b = laplaceianReduce(img_2, levels)
    g_m = gaussianPyr(mask, levels)
    ans = (lapla_a[levels - 1] * g_m[levels - 1] + (1 - g_m[levels - 1]) * lapla_b[levels - 1])
    gs_k = get_gaussian() * 4
    k = levels - 2
    while k >= 0:  # go through the list from end to start
        ans = gaussExpand(ans, gs_k) + lapla_a[k] * g_m[k] + (1 - g_m[k]) * lapla_b[k]
        k -= 1
    naive_blend = cv2.resize(naive_blend, (ans.shape[1], ans.shape[0]))
    return naive_blend, ans


def resizemask(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
    new_width = mask.shape[1]
    new_height = mask.shape[0]
    img1 = cv2.resize(img1, (new_width, new_height))
    img2 = cv2.resize(img2, (new_width, new_height))
    return img1, img2


def naive_blending(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ans = img1 * mask + img2 * (1 - mask)
    return ans


def get_gaussian():
    kernel = cv2.getGaussianKernel(5, -1)
    kernel = kernel.dot(kernel.T)
    return kernel


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    if img.ndim == 3:  # RGB img
        padded_im = np.zeros(((img.shape[0] * 2), (img.shape[1] * 2), 3))
    else:
        padded_im = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    padded_im[::2, ::2] = img
    return cv2.filter2D(padded_im, -1, gs_k)
