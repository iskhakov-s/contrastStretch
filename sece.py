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
hist - a dictionary of hists, keys are tuples (1, 1) to (M,N)
entropy - an ndarray of length 256 w the entropy of each pixel intensity
H, W - the height and width of the image, as ints
"""

def spatial_hist_entropy(img):
    H, W = img.shape
    r = H / W
    K = 256
    
    M = round(math.sqrt(K*r))
    N = round(math.sqrt(K/r))
    
    hist = {(m+1,n+1):None for m in range(M) for n in range(N)}
    entropy = np.zeros(256)
    
    for m in range(1,M+1):
        for n in range(1,N+1):
            region = img[(m-1) * H / M : m * H / M, (n-1) * W / N : n * W / N]
            hist[(m,n)] = cv.calcHist([region],[0],None,[256],[0,256])
            entropy -= hist[(m,n)] * np.log2(hist[(m,n)])
            
    return hist, entropy, H, W


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


def dct(img):
    img_f = np.float32(img)
    dst = cv2.dct(img_f)
    img = np.uint8(dst)
    
    return img


def idct(img):
    img_f = np.float32(img)
    dst = cv2.idct(img_f)
    img = np.uint8(dst)
    
    return img
    

# probably not optimal
def weighting_coefficient(discrete_func,H,W, gamma = 0.5):
    alpha = sum([i*math.log(i,2) for i in discrete_func]) ** gamma
    
    temp1 = (alpha - 1) / (H - 1)
    temp2 = (alpha - 1) / (W - 1)
    
    w = np.zeros((H,W))
    for k in range(H):
        for l in range(W):
            w[k][l] = (1+temp1*k) * (1+temp2*l)
    
    return w


def sece(img):
    hist, s, H, W = spatial_hist_entropy(img)
    f = discrete_func(s)
    cdf = np.cumsum(f)
    out = mapping(img, cdf)
    
    return out, f, H, W


"""
Applies SECE w Discrete Cosine Transform
First SECE, then DCT, then applies a weighting coefficent, then idct

parameter:
img - a grayscale image or the v channel of an hsv color image, 2D

returns:
img_idct - a grayscale image or the v channel of an hsv color image, 2D
"""
def sece_dct(img):
    img_sece, f, H, W = sece(img)
    img_dct = dct(img_sece)
    w = weighting_coefficient(f,H,W)
    img_w = img_dct * w
    img_idct = idct(img_w)
    
    return img_idct