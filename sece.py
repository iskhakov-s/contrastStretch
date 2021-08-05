# Spatial Entropy-Based Contrast Enhancement (w Discrete Cosine Transform)
# the img parameter in all functions is either grayscale or the v channel of a color image
import cv2
import numpy as np
import math


"""
Divide the image into M*N regions, and calculate the hist for each
Then find the spatial entropy using -xlog2(x)
Combined in one function to minimize runtime

parameter:
img - a grayscale image or the v channel of an hsv color image, 2D

returns:
entropy - an ndarray of length 256 w the entropy of each pixel intensity
H, W - the height and width of the image, as ints
"""

def spatial_hist_entropy(img):
    H, W = img.shape
    r = H / W
    K = 256
    
    M = round(math.sqrt(K*r))
    N = round(math.sqrt(K/r))
    
    # hist = {(m+1,n+1):None for m in range(M) for n in range(N)}
    entropy = np.zeros((256))
    
    for m in range(1,M+1):
        for n in range(1,N+1):
            s_x1 = round((m-1) * H / M)
            s_x2 = round(m * H / M)
            s_y1 = round((n-1) * W / N)
            s_y2 = round(n * W / N)
            region = img[s_x1 : s_x2, s_y1 : s_y2]
            # hist[(m,n)] = cv2.calcHist([region],[0],None,[256],[0,256])
            h = cv2.calcHist([region],[0],None,[256],[0,256]).flatten()
            # entropy -= np.where(hist[(m,n)] > 0.0, hist[(m,n)] * np.log2(hist[(m,n)]), 0)
            # entropy -= np.where(h > 0.0, h * np.log2(h), 0)
            entropy += h * np.log2(h, where = h>0)
            
    return entropy, H, W


def discrete_func(entropy):
    total = np.sum(entropy)
    
    func = entropy / (total - entropy)
    func = func / np.sum(func)
    
    return func
    # for cdf, use np.cumsum()
    

def mapping(img, cdf):
    mapper = lambda k : 255 * cdf[k]
    out = mapper(img)
    return out


# def dct(img):
#     img_f = np.float32(img)
#     dst = cv2.dct(img_f)
#     np.putmask(dst, dst > 255, 255)
#     np.putmask(dst, dst < 0, 0)
#     img = np.uint8(dst)
    
#     return img


# def idct(img):
#     img_f = np.float32(img)
#     dst = cv2.idct(img_f)
#     np.putmask(dst, dst > 255, 255)
#     np.putmask(dst, dst < 0, 0)
#     img = np.uint8(dst)
    
#     return img
    

# probably not optimal
def weighting_coefficient(discrete_func,H,W, gamma = 0.5):
    # alpha = sum([k*math.log2(k) if k>0 else k for k in discrete_func]) ** gamma
    alpha = np.sum(-discrete_func * np.log2(discrete_func, where = discrete_func>0)) ** gamma
    
    temp1 = (alpha - 1) / (H - 1)
    temp2 = (alpha - 1) / (W - 1)
    
    w = np.zeros((H,W))
    for k in range(H):
        for l in range(W):
            w[k,l] = (1+temp1*k) * (1+temp2*l)
    
    return w


def sece(img, forDCT = False):
    img = np.float32(img)
    s, H, W = spatial_hist_entropy(img)
    f = discrete_func(s)
    cdf = np.cumsum(f)
    img = np.uint8(img)
    out = mapping(img, cdf)
    
    if forDCT:
        return out, f, H, W
    
    np.putmask(out, out > 255, 255)
    np.putmask(out, out < 0, 0)
    out = np.uint8(out)
    return out


"""
Applies SECE w Discrete Cosine Transform
First SECE, then DCT, then applies a weighting coefficent, then idct

parameter:
img - a grayscale image or the v channel of an hsv color image, 2D

returns:
img_idct - a grayscale image or the v channel of an hsv color image, 2D
"""
def sece_dct(img):
    img_sece, f, H, W = sece(img, forDCT = True)
    img_f = np.float32(img_sece)
    img_dct = cv2.dct(img_f)
    w = weighting_coefficient(f,H,W)
    img_w = img_dct * w
    img_f = np.float32(img_w)
    img_idct = cv2.idct(img_f)

    np.putmask(img_idct, img_idct > 255, 255)
    np.putmask(img_idct, img_idct < 0, 0)
    out = np.uint8(img_idct)
    
    return out