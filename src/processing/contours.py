"""
implementation of various contour extraction methods
"""
import sys
sys.path.append('..')

import numpy as np
import cv2
from features.preprocessing import get_sobel


def _get_small_contours(mask):
    kernel = np.ones((3,3),np.uint8)
    sure_fg = cv2.erode(mask, kernel, iterations=1)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    return unknown

def _get_big_contours(mask):
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_fg = cv2.dilate(mask,
                        kernel=kernel_d,
                        iterations=5)
    sure_bg = cv2.erode(sure_fg, 
                         kernel=kernel_e,
                         iterations=3)
    unknown = cv2.subtract(sure_fg, sure_bg)
    return unknown

def get_contours(bubble_size, mask):
    if bubble_size=='big':
        return _get_big_contours(mask)
    elif bubble_size=='small':
        return _get_small_contours(mask)
    

def get_sobel_contours(img: np.array, markers):
    invmask = 255 - markers

    grad = get_sobel(img, scale=1, delta=0, ksize=3, ddepth=cv2.CV_16S)
    _, tr = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dtr = cv2.morphologyEx(tr, cv2.MORPH_ERODE, kernel)
    contours = (dtr * invmask)
    return contours


def get_morph_contours(img: np.array):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphgradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    _, mgtr = cv2.threshold(morphgradient, 50, 255, cv2.THRESH_BINARY)
    emgtr = cv2.morphologyEx(mgtr, cv2.MORPH_DILATE, kernel)
    return morphgradient


def get_laplace_contours(img: np.array):
    lapl = cv2.Laplacian(img, cv2.CV_16S, ksize=3, scale=2)
    l = cv2.convertScaleAbs(lapl)
    return l


def get_laplace_morph_contours(img: np.array):
    l = get_laplace_contours(img)
    m = get_morph_contours(l)
    return m