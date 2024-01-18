import numpy as np

def get_mu1(hist: np.array, t: int):
    return np.average(hist[:t], weights=hist[:t])

def get_omega1(hist: np.array, t:int):
    return np.sum(hist[:t])

def get_mu2(hist: np.array, t: int):
    return np.average(hist[t:], weights=hist[t:])

def get_omega2(hist: np.array, t: int):
    return np.sum(hist[t:])

def get_var1(hist, t):
    om1 = get_omega1(hist, t)
    return om1*get_mu1(hist, t)**2 if om1!=0 else om1

def get_var2(hist, t):
    om2 = get_omega2(hist, t)
    return om2*get_mu2(hist, t)**2 if om2!=0 else om2

def get_weighted_otsu_threshold(img: np.array, hist_range=255):
    """
    a method for searching segmentation threshold \
    using weighted otsu method
    args:
        img: source image
        hist_range: max range for hist bins.
    """
    T = np.arange(0, hist_range+1, step=1)
    hist, _ = np.histogram(img.ravel(), bins=T, density=True)
    thresh_array = [
            (1-hist[t])*(get_var1(hist, t) + get_var2(hist, t))
            for t in T[:-1]]

    thresh = np.argmax(thresh_array)
    return thresh, thresh_array

