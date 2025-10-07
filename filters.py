import cv2
import numpy as np
from typing import Union, List, Tuple

# parte de limiarização 
def thresholding_segmentation(img_cv, *args, threshold_value: int = 90, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

def otsu_segmentation(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

#  parte de morfologia 
def erosion(img_cv, *args, kernel_size: int = 5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    k = _square_kernel(kernel_size)
    eroded = cv2.erode(gray_img, k, iterations=2)
    return cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)

def dilatation(img_cv, *args, kernel_size: int = 5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    k = _square_kernel(kernel_size)
    dil = cv2.dilate(gray_img, k, iterations=1)
    return cv2.cvtColor(dil, cv2.COLOR_GRAY2BGR)

def open(img_cv, *args, kernel_size: int = 5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Otsu 
    _, seg = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = _square_kernel(kernel_size)
    opened = cv2.morphologyEx(seg, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

def close(img_cv, *args, kernel_size: int = 5, **kwargs):
    if img_cv is None:
        return None
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, seg = cv2.threshold(gray_img, 90, 255, cv2.THRESH_BINARY)
    k = _square_kernel(kernel_size)
    closed = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, k, iterations=1)
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

# suavização básica
def low_pass(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    return cv2.GaussianBlur(img_cv, (15, 15), 0)

def low_pass_gaussian(img_cv, *args, sigma: Union[float, int] = 3.0, filter_shape: Union[List, Tuple, None] = None, **kwargs):
    if img_cv is None:
        return None
    if filter_shape is None:
        size = 2 * int(4 * float(sigma) + 0.5) + 1  # tamanho ímpar ~ 8*sigma
        filter_shape = (size, size)
    kx = cv2.getGaussianKernel(filter_shape[0], sigma)
    ky = cv2.getGaussianKernel(filter_shape[1], sigma)
    g = (kx @ ky.T).astype(np.float32)
    return cv2.filter2D(img_cv, -1, g)

def low_pass_media(img_cv, *args, kernel_size: int = 9, **kwargs):
    if img_cv is None:
        return None
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(img_cv, -1, kernel)

# realce das bordas
def high_pass(img_cv, *args, **kwargs):
    if img_cv is None:
        return None
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def high_pass_laplacian(img_cv, *args, kernel_value: int = 3, **kwargs):
    if img_cv is None:
        return None
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    ksize = kernel_value if kernel_value % 2 == 1 else kernel_value + 1
    kernel = np.ones((ksize, ksize), dtype=np.float32) * -1
    kernel[ksize // 2, ksize // 2] = ksize * ksize - 1
    filtered = cv2.filter2D(gray, -1, kernel)
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

def high_pass_sobel(img_cv, *args, direction: str = 'x', **kwargs):
    if img_cv is None:
        return None
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    if direction == 'y':
        kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
    else:
        kernel = np.array([[-1,  0,  1],
                           [-2,  0,  2],
                           [-1,  0,  1]], dtype=np.float32)
    sob = cv2.filter2D(gray, cv2.CV_64F, kernel)
    sob = cv2.convertScaleAbs(sob)
    return cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)

def _square_kernel(k: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return np.ones((k, k), np.uint8)


# kmeans
def kmeans_segmentation(img_cv, *args, k: int = 3, **kwargs):
    
    if img_cv is None:
        return None
    Z = img_cv.reshape((-1, 3)).astype(np.float32)  # N x 3 float32
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(Z, int(k), None, criteria, 10, flags)
    centers = np.uint8(centers)
    seg = centers[labels.flatten()].reshape(img_cv.shape)
    return seg  

def kmeans_segmentation_mask(img_cv, *args, k: int = 3, **kwargs):
    
    if img_cv is None:
        return None
    seg = kmeans_segmentation(img_cv, k=k)
    if seg is None:
        return None
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
