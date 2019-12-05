import cv2 
import numpy as np
def rgb2lab(src):
    r = src[0, 0, :, :]
    g = src[0, 1, :, :]
    b = src[0, 2, :, :]
    rgb = (cv2.merge([r, g, b]))

    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    return lab

def lab2rgb(src):
    l = src[0, 0, :, :]
    a = src[0, 1, :, :]
    b = src[0, 2, :, :]

    lab = cv2.merge([l, a, b])

    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return rgb

def rgb2hsv(src):
    r = src[0, 0, :, :]
    g = src[0, 1, :, :]
    b = src[0, 2, :, :]
    rgb = (cv2.merge([r, g, b]))

    hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    return hsv

def hsv2rgb(src):
    h = src[0, 0, :, :]
    s = src[0, 1, :, :]
    v = src[0, 2, :, :]

    hsv = cv2.merge([h, s, v])

    rgb = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
    return rgb

def rgb2ycbcr(src):
    r = src[0, 0, :, :]
    g = src[0, 1, :, :]
    b = src[0, 2, :, :]
    rgb = (cv2.merge([r, g, b]))

    ycbcr = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    return ycbcr

def ycbcr2rgb(src):
    y = src[0, 0, :, :]
    cb = src[0, 1, :, :]
    cr = src[0, 2, :, :]

    ycbcr = cv2.merge([y, cb, cr])

    rgb = cv2.cvtColor(ycbcr.astype(np.float32), cv2.COLOR_YCrCb2RGB)
    return rgb

def rgb2yuv(src):
    r = src[0, 0, :, :]
    g = src[0, 1, :, :]
    b = src[0, 2, :, :]
    rgb = (cv2.merge([r, g, b]))

    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LUV)
    return lab

def yuv2rgb(src):
    l = src[0, 0, :, :]
    u = src[0, 1, :, :]
    v = src[0, 2, :, :]

    luv = cv2.merge([l, u, v])

    rgb = cv2.cvtColor(luv.astype(np.float32), cv2.COLOR_LUV2RGB)
    return rgb
    