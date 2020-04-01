import numpy as np
import math
import cv2
from scipy.signal import convolve2d
from scipy import ndimage

# load once
from pathlib import Path
cascPath=Path(__file__).parent.absolute()/"haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(str(cascPath))

def get_face_num(gray):
    """
    # %%
    from get_features import *
    import cv2
    img = cv2.imread('D:/temp_videos/WIN_20190728_23_21_24_Pro.jpg', 0) # 0: gray
    get_face_num(img)
    # %%
    """
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    return len(faces)

def get_colorfulness(image):
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rgMean, rgStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def get_brightness(gray_img):
    return np.mean(gray_img)

def get_contrast(gray_img):
    lumi_avg = np.mean(gray_img)
    new_matrix = gray_img-lumi_avg
    new_matrix = np.square(new_matrix)
    ms=np.mean(new_matrix)
    rms=ms**0.5
    return rms

def get_gaussian_spatial(I):
    # X scale 1
    f = ndimage.filters.gaussian_filter1d(I, 1, axis=0, order=1, output=None, mode='constant', cval=0.0)
    Fx_1 = ndimage.filters.gaussian_filter1d(f, 3, axis=1, order=0, output=None, mode='constant', cval=0.0)
    #Fx_1 = 255*(Fx_1 - np.min(Fx_1))/(np.max(Fx_1))

    # X scale 2
    f = ndimage.filters.gaussian_filter1d(I, 2, axis=0, order=1, output=None, mode='constant', cval=0.0)
    Fx_2 = ndimage.filters.gaussian_filter1d(f, 6, axis=1, order=0, output=None, mode='constant', cval=0.0)

    # X scale 3
    f = ndimage.filters.gaussian_filter1d(I, 4, axis=0, order=1, output=None, mode='constant', cval=0.0)
    Fx_3 = ndimage.filters.gaussian_filter1d(f, 12, axis=1, order=0, output=None, mode='constant', cval=0.0)

    # Y scale 1
    f = ndimage.filters.gaussian_filter1d(I, 1, axis=1, order=1, output=None, mode='constant', cval=0.0)
    Fy_1 = ndimage.filters.gaussian_filter1d(f, 3, axis=0, order=0, output=None, mode='constant', cval=0.0)

    # Y scale 2
    f = ndimage.filters.gaussian_filter1d(I, 2, axis=1, order=1, output=None, mode='constant', cval=0.0)
    Fy_2 = ndimage.filters.gaussian_filter1d(f, 6, axis=0, order=0, output=None, mode='constant', cval=0.0)

    # Y scale 3
    f = ndimage.filters.gaussian_filter1d(I, 4, axis=1, order=1, output=None, mode='constant', cval=0.0)
    Fy_3 = ndimage.filters.gaussian_filter1d(f, 12, axis=0, order=0, output=None, mode='constant', cval=0.0)

    features_x = [  np.mean(abs(Fx_1)),
                    np.mean(abs(Fx_2)),
                    np.mean(abs(Fx_3))]
    features_y = [  np.mean(abs(Fy_1)),
                    np.mean(abs(Fy_2)),
                    np.mean(abs(Fy_3))]
    return features_x, features_y

def get_gaussian_temporal(V):
    [K,M,N] = V.shape

    V1 = ndimage.filters.gaussian_filter1d(V, 2, axis = 0, order= 1, output=None, mode='reflect', cval=0.0, truncate=4.0)
    V2 = ndimage.filters.gaussian_filter1d(V, 4, axis = 0, order= 1, output=None, mode='reflect', cval=0.0, truncate=4.0)
    V3 = ndimage.filters.gaussian_filter1d(V, 8, axis = 0, order= 1, output=None, mode='reflect', cval=0.0, truncate=4.0)

    mean_v1 = np.mean(V1, axis =0)
    mean_v2 = np.mean(V2, axis =0)
    mean_v3 = np.mean(V3, axis =0)
    features_t = [
        [np.mean(mean_v1), np.std(mean_v1)],
        [np.mean(mean_v2), np.std(mean_v2)],
        [np.mean(mean_v3), np.std(mean_v3)],
    ]

    return features_t
