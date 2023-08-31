from matplotlib import pyplot as plt
from typing import List

import cv2
import numpy as np
from numpy.linalg import LinAlgError

from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float_)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(np.float_), img_2.astype(np.float_), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)

def hierarchicalkDemo(img_path):
    print("---------------------------------------------------------------------------")
    print("Hierarchical LK Demo")
    im1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float_)
    im2 = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

    st = time.time()
    ans = opticalFlowPyrLK(im1.astype(np.float_), im2.astype(np.float_), 4, 20, 5)
    et = time.time()

    print("Time: {:.2f}".format(et - st))
    pts = np.array([])
    uv = np.array([])
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            if ans[i][j][1] != 0 and ans[i][j][0] != 0:
                uv = np.append(uv, ans[i][j][0])
                uv = np.append(uv, ans[i][j][1])
                pts = np.append(pts, j)
                pts = np.append(pts, i)
    pts = pts.reshape(int(pts.shape[0] / 2), 2)
    uv = uv.reshape(int(uv.shape[0] / 2), 2)
    print(np.median(uv, 0))
    print(np.mean(uv, 0))
    displayOpticalFlow(im2, pts, uv)


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=np.float_)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    pts1, uv1 = opticalFlow(img_1.astype(np.float_), img_2.astype(np.float_), step_size=20, win_size=5)
    arr3d = opticalFlowPyrLK(img_1.astype(np.float_), img_2.astype(np.float_), stepSize=20, winSize=5, k=5)
    pts2, uv2=arr3d2pts(arr3d)
    displaydiffrents(img_2, pts1, uv1 ,img_2, pts2, uv2)

def arr3d2pts(mat:np.ndarray) ->(np.ndarray, np.ndarray):
    x_y_return = []
    u_v_return = []
    for x in range(len(mat)):
        for y in range(len(mat[0])):
            dummy = mat[x, y]
            if mat[x, y][0] != 0 or mat[x, y][1] != 0:
                x_y_return.append([y, x])
                u_v_return.append(mat[x, y])
    return np.array(x_y_return), np.array(u_v_return)
def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()
    print("---------------------------------------------------------------------------")
    print("Optical flow demo")

def displaydiffrents(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray,img2: np.ndarray, pts2: np.ndarray, uvs2: np.ndarray):
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')
    ax[0].set_title('lkDemo image')
    ax[1].imshow(img2, cmap='gray')
    ax[1].quiver(pts2[:, 0], pts2[:, 1], uvs2[:, 0], uvs2[:, 1], color='r')
    ax[1].set_title('hierarchicalkDemo image')
    plt.show()
    cv2.waitKey(0)



# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def TranslationLK(img_path):
    print("translation LK")
    img_path = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_path = cv2.resize(img_path, (0, 0), fx=.5, fy=0.5)
    cv2.imwrite('imTransA1.jpg', cv2.cvtColor(img_path.astype(np.uint8), cv2.COLOR_RGB2BGR))
    t = np.array([[1, 0, -5],
                  [0, 1, -5],
                  [0, 0, 1]], dtype=np.float_)
    img_2 = cv2.warpPerspective(img_path, t, img_path.shape[::-1])
    mat = findTranslationLK(img_path, img_2)
    img_3 = cv2.warpPerspective(img_path, mat, img_path.shape[::-1])
    cv2.imshow("translation from cv2", img_2)
    cv2.imshow("translation LK me", img_3)
    cv2.imwrite('imTransA2.jpg', cv2.cvtColor(img_2.astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(mat)
    cv2.waitKey(0)


def TranslationCorrelation(img_path):
    print("translation correlation")
    img_path = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_path = cv2.resize(img_path, (0, 0), fx=.5, fy=0.5)
    cv2.imwrite('imTransB1.jpg', cv2.cvtColor((img_path).astype(np.uint8), cv2.COLOR_RGB2BGR))
    t = np.array([[1, 0, -20],
                  [0, 1, -20],
                  [0, 0, 1]], dtype=np.float_)
    img_2 = cv2.warpPerspective(img_path, t, img_path.shape[::-1])
    mat = findTranslationCorr(img_path, img_2)
    img_3 = cv2.warpPerspective(img_path, mat, img_path.shape[::-1])
    cv2.imshow("translation from cv2", img_2)
    cv2.imshow("translation Correlation me", img_3)
    cv2.imwrite('imTransB2.jpg', cv2.cvtColor((img_2).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(mat)
    cv2.waitKey(0)




def RigidLK(img_path):
    print("Rigid LK ")
    #

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    cv2.imwrite('imRigidA1.jpg', cv2.cvtColor((img_1).astype(np.uint8), cv2.COLOR_RGB2BGR))
    t = np.array([[np.cos(np.radians(0.6)), -np.sin(np.radians(0.6)), -0.5],
                  [np.sin(np.radians(0.6)), np.cos(np.radians(0.6)), -0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float_)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    # cv2.imshow("Rigid", img_2)

    matrix = findRigidLK(img_1, img_2)
    img_3 = cv2.warpPerspective(img_1, matrix, img_1.shape[::-1])
    cv2.imshow("Rigid from cv2", img_2)
    cv2.imshow("RigidLK from me", img_3)
    print(matrix)
    cv2.waitKey(0)

    cv2.imwrite('imRigidA2.jpg', cv2.cvtColor((img_2).astype(np.uint8), cv2.COLOR_RGB2BGR))


def RigidCorrelation(img_path):
    print("Rigid Correlation ")
    #

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    cv2.imwrite('imRigidB1.jpg', cv2.cvtColor((img_1).astype(np.uint8), cv2.COLOR_RGB2BGR))
    t = np.array([[np.cos(np.radians(0.6)), -np.sin(np.radians(0.6)), 5],
                  [np.sin(np.radians(0.6)), np.cos(np.radians(0.6)), 6],
                  [0.0, 0.0, 1.0]], dtype=np.float_)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    # cv2.imshow("Rigid", img_2)

    matrix = findRigidCorr(img_1, img_2)
    img_3 = cv2.warpPerspective(img_1, matrix, img_1.shape[::-1])
    cv2.imshow("Rigid from cv2", img_2)
    cv2.imshow("RigidCorr from me", img_3)
    print(matrix)
    cv2.waitKey(0)

    cv2.imwrite('imRigidB2.jpg', cv2.cvtColor((img_2).astype(np.uint8), cv2.COLOR_RGB2BGR))




def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")
    img_path = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_path = cv2.resize(img_path, (0, 0), fx=.5, fy=0.5)
    t = np.array([[0.9, 0, 0],
                  [0, 0.9, 0],
                  [0, 0, 1]], dtype=np.float_)
    img_2 = cv2.warpPerspective(img_path, t, img_path.shape[::-1])
    my_function = warpImages(img_path, img_2, t)
    # cv2.imshow("original photo", img_path)
    # cv2.imshow("warp from me", my_function)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img_path)
    ax[0].set_title('original photo')
    ax[1].imshow(img_2)
    ax[1].set_title('warped')
    ax[2].imshow(my_function)
    ax[2].set_title('my inverse warp')
    plt.show()
    cv2.waitKey(0)




# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset .jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat .jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)
    imageWarpingDemo(img_path)
    img_path = 'input/sunset .jpg'
    #TranslationLK(img_path) #not working
    img_path = 'input/sunset .jpg'
    TranslationCorrelation(img_path)
    # Rigid
    img_path = 'input/pyr_bit.jpg'
    #RigidLK(img_path) #not woriking
    img_path = 'input/pyr_bit.jpg'
    #RigidCorrelation(img_path)

    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo() # need to fill


if __name__ == '__main__':
    main()
