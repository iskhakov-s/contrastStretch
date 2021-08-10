import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power

threshold = 0.3
T_t = 112


def image_agcwd(img, a=0.25, truncated_cdf=False):
    h,w = img.shape[:2]
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)
    intensity_max = unique_intensity.max()
    intensity_min = unique_intensity.min()
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()
    
    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp>0] = prob_max * (pn_temp[pn_temp>0]**a)
    pn_temp[pn_temp<0] = prob_max * (-((-pn_temp[pn_temp<0])**a))
    prob_normalized_wd = pn_temp / pn_temp.sum() # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
    
    if truncated_cdf: 
        inverse_cdf = np.maximum(0.5,1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd
    
    img_new = img.copy()
    for i in unique_intensity:
        img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i])
   
    return img_new


def process_bright(img):
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reverse = 255 - agcwd
    return reverse


def process_dimmed(img):
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)
    return agcwd


# method assumes only a brightness channel, like v in hsv, is passed
def iagcwd(img):
    M,N = img.shape
    m_I = np.sum(img/(M*N)) 
    t = (m_I - T_t)/ T_t
    
    if t < -threshold:
        result = process_dimmed(img)
        img = result
    elif t > threshold:
        result = process_bright(img)
        img = result
    print('img not suitable for iagcwd')
    return img


def agcwd(img, hist):
    l_max = 256
    pdf = hist.flatten() / img.size
    pdf_max = pdf.max()
    pdf_min = pdf.min()
    alpha = 0.5
    pdf_w = pdf_max * ( (pdf - pdf_min) / (pdf_max - pdf_min) ) ** alpha

    gamma = 1 - np.cumsum(pdf_w) / np.sum(pdf_w)
    transformation = lambda l : l_max * (l / l_max) ** gamma[l]
    out = transformation(img)
    np.putmask(out, out > 255, 255)
    np.putmask(out, out < 0, 0)
    out = np.uint8(out)
    
    return out