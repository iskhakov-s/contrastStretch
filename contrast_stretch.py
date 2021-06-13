import numpy as np
import cv2


m1_bounds = 5
m2_cutoff = 0.05
algo_a = 0
algo_b = 255

# method 1
# find the pixel val that contains a specific amount of data below it
def m1_weighted_percentile(hist, percentile):
    cdf = hist.cumsum()
    thresh = cdf[-1] * percentile / 100
    # binary search, which finds the smallest index >= to the threshold
    low = 0
    high = len(cdf) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        if cdf[mid] < thresh:
            low = mid + 1
        elif cdf[mid] > thresh:
            high = mid - 1
        else:
            break
    return mid


# method 2
# finds the lowest and highest vals in the histogram that is greater than 
#     a percentage of the peak val
def m2_percent_distribution(hist):
    peak = np.amax(hist)
    peak_index = hist.tolist().index(peak)
    cutoff = m2_cutoff * peak
    c = 0; d = 255
    for i in range(peak_index):
        if hist[i] > cutoff:
            c = i
            break
    for i in range(np.size(hist)-1, peak_index, -1):
        if hist[i] > cutoff:
            d = i
            break
    return c, d


# the contrast stretch algorithm
def algorithm(c, d, img):
    a = algo_a; b = algo_b
    scalar = (b-a)/(d-c)
    algo_applied = np.add(img, - c)
    algo_applied = np.multiply(algo_applied, scalar)
    algo_applied = np.add(algo_applied, a)

    np.putmask(algo_applied, algo_applied > 255, 255)
    np.putmask(algo_applied, algo_applied < 0, 0)
    algo_applied = algo_applied.astype("uint8")
    return algo_applied


# applies m1/m2 stretch and algorithm methods
def apply_stretch(hist, img):
    c1 = m1_weighted_percentile(hist, m1_bounds)
    d1 = m1_weighted_percentile(hist, 100-m1_bounds)
    m1_img = algorithm(c1, d1, img)

    c2, d2 = m2_percent_distribution(hist)
    m2_img = algorithm(c2, d2, img)

    return m1_img, m2_img