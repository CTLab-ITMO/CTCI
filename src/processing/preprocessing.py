import cv2
import numpy as np
import skimage.morphology as morph

# the data is homogeneous & specific,
# preprocesses are edge extraction and segmentation

def get_edges(img, low_thresh, high_thresh):
    #convert to grayscale and use canny
    grayscale = cv2.cvtColor(img, cv2.BGR2GRAY)
    return cv2.Canny(grayscale, low_thresh, high_thresh)

def single_scale_retinex(img, sigma):
    """
    single scale retinex algorithm

    args:
        img: image to process
        sigma: weight of processing. empirically selected
    """
    return np.log(img) - np.log(cv2.GaussianBlur(img, (0,0), sigma))
 
 
def multi_scale_retinex(img, scales, sigma_list):
    """
    multi scale retinex algorithm

    it is a weighted sum of multiple results of ssr with different sigma's

    args:
        img: image to process
        scales: weights of sum of the result. sum of scales is assumed to be equal 1
        sigma_list: sigma's for ssr
    """
    result = np.zeros_like(img)

    assert len(scales) == len(sigma_list), "list of scales and list of sigmas should be the same length"
    assert sum(scales) == 1, "sum of scales should be equal to 1"

    for scale, sigma in zip(scales, sigma_list):
        result = result + scale * single_scale_retinex(img, sigma)
    return result


def bilateral_filtering(img, diameter, sigma_color, sigma_space):
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

def morphological_top_hat(gray_img, kernel_size=(3,3)):
    """
    wrapper for morphological top hat operation

    args:
        gray_img: grayscale image
        kernel_size: size of filtering kernel
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)

def morphological_bottom_hat(gray_img, kernel_size=(3,3)):
    """
    wrapper for morphological bottom hat operation

    args:
        gray_img: grayscale image
        kernel_size: size of filtering kernel
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

def morphological_transform(img):
    """
    morphological transform used to extract foreground markers from bubble image

    args:
        img: image to process
    """
    img_open = morphological_top_hat(img)
    img_closed = morphological_bottom_hat(img)

    img_0 = img - img_open + img_closed

    return img_0
