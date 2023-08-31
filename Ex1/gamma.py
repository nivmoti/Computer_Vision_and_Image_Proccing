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
import cv2
import numpy as np

from ex1_utils import LOAD_GRAY_SCALE

global imgGamma


def gammaDisplay(img_path: str, rep: int) -> None:
    global img
    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img/255
    cv2.namedWindow("Gamma Image")
    cv2.createTrackbar("Gamma", "Gamma Image", 1, 200, onTrackBar)
    cv2.imshow("Gamma Image", img)
    onTrackBar(100)
    cv2.waitKey()


def onTrackBar(val):
    global img
    val = val/100
    newImg = np.power(img, val) # apply the gamma algoritham on a image
    cv2.imshow("Gamma Image", newImg)

def main():
    gammaDisplay('../Ex1/beach.jpg', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
