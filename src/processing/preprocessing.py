import cv2
import numpy as np
import skimage.morphology as morph

# the data is homogeneous & specific,
# preprocesses are edge extraction and segmentation


def minmax(X, range=[0,1]):
    std = (X - X.min()) / (X.max() - X.min())
    scaled = std * (range[1] - range[0]) + range[0]
    return scaled.astype('uint8')


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)    
    for i in arr:
        temp = (((i - np.min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr)
 

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)

def morphological_bottom_hat(gray_img, kernel_size=(3,3)):
    """
    wrapper for morphological bottom hat operation

    args:
        gray_img: grayscale image
        kernel_size: size of filtering kernel
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)


def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
    """
    Iteratively expand the markers white keeping them limited by the mask during each iteration.

    args:
        marker: Grayscale image where initial seed is white on black background.
        mask: Grayscale mask where the valid area is white on black background.
        radius Can be increased to improve expansion speed while causing decreased isolation from nearby areas. 
    
    returns:
        A copy of the last expansion.

    Written By Semnodime.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded

def morphological_transform(img):
    """
    morphological transform used to extract foreground markers from bubble image

    args:
        img: image to process
    """
    img_open = morphological_top_hat(img)
    img_closed = morphological_bottom_hat(img)

    img_0 = img - img_open + img_closed

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    g_0 = cv2.morphologyEx(img_0, cv2.MORPH_OPEN, kernel)
    f_0c = imreconstruct(img_0, g_0)

    g_0c = cv2.morphologyEx(f_0c, cv2.MORPH_CLOSE, kernel)
    f_0cr = imreconstruct(f_0c, g_0c)

    return f_0cr


def get_sobel(image, scale, ksize, delta, ddepth):
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_CONSTANT)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_CONSTANT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
        
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def preprocess(img: np.array):
    ssr_img = single_scale_retinex(img, 80).astype('float32')
    bf = bilateral_filtering(ssr_img, 5, 75, 75)
    morphed = morphological_transform(bf)
    morphed = cv2.GaussianBlur(morphed, (5,5), 0)
    m = minmax(morphed.ravel(), [0, 255]).astype('uint8')
    i = m.reshape(img.shape)
    return i