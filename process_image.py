import numpy as np
import cv2
from matplotlib import pyplot as plt

class ProcessImg:
    
    m1_bounds = 5
    m2_cutoff = 0.05
    algo_a = 0
    algo_b = 255
    hists = {}
    new_img = {}

    #TODO: replace 'self' in function arguments with None
    #      then check in the func if the arg is not defined set it to the thing you want
    #      https://stackoverflow.com/questions/1802971/nameerror-name-self-is-not-defined
    
    #TODO: can be expanded by having the functions implicitly work regardless of # of channels
    #      or make subclass for rgb and greyscale images 
    
    def __init__(self, img, rgb = True):
        self.img = img
        if rgb:
            # make the rgb hists, stretch each channel, combine and make hists for stretched
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            r, g, b = cv2.split(self.img)
            self.hists['r'], self.hists['g'], self.hists['b'] = self.make_hists()
            self.stretched_r1, self.stretched_r2 = self.apply_stretch(self.hists['r'], r)
            self.stretched_g1, self.stretched_g2 = self.apply_stretch(self.hists['g'], g)
            self.stretched_b1, self.stretched_b2 = self.apply_stretch(self.hists['b'], b)
            self.new_img['stretch_m1'] = cv2.merge((self.stretched_r1, self.stretched_g1, self.stretched_b1))
            self.new_img['stretch_m2'] = cv2.merge((self.stretched_r2, self.stretched_g2, self.stretched_b2))
            self.hists['r_s1'], self.hists['g_s1'], self.hists['b_s1'] = self.make_hists(self.new_img['stretch_m1'])
            self.hists['r_s2'], self.hists['g_s2'], self.hists['b_s2'] = self.make_hists(self.new_img['stretch_m2'])
        else:
            # make the grey hist, stretch it, and make a new hist
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.hists['bw'] = cv2.calcHist([self.img], [0], None, [256], [0,256])
            self.new_img['stretch_m1'], self.new_img['stretch_m2'] = self.apply_stretch(self.hists['bw'])
            self.hists['bw_s1'] = cv2.calcHist([self.new_img['stretch_m1']], [0], None, [256], [0,256])
            self.hists['bw_s2'] = cv2.calcHist([self.new_img['stretch_m2']], [0], None, [256], [0,256])
        
        # creates new images using equalization
        self.new_img['equalization_m1'] = self.hist_equalization_1()
        self.new_img['equalization_m2'] = self.hist_equalization_2()
        
        # makes new hists based on the equalized images
        if rgb:
            self.hists['r_e1'], self.hists['g_e1'], self.hists['b_e1'] = self.make_hists(self.new_img['equalization_m1'])
            self.hists['r_e2'], self.hists['g_e2'], self.hists['b_e2'] = self.make_hists(self.new_img['equalization_m2'])
        else:
            self.hists['bw_e1'] = cv2.calcHist([self.new_img['equalization_m1']], [0], None, [256], [0,256])
            self.hists['bw_e2'] = cv2.calcHist([self.new_img['equalization_m2']], [0], None, [256], [0,256])

            
    # takes an image and makes a hist for each color channel
    # should only be called for rgb (or other 3-channel) images
    def make_hists(self, img = None):
        if img is None:
            img = self.img
        
        r, g, b = cv2.split(img)
        r = cv2.calcHist([r], [0], None, [256], [0,256])
        g = cv2.calcHist([g], [0], None, [256], [0,256])
        b = cv2.calcHist([b], [0], None, [256], [0,256])
        return r, g, b
    
    # method 1
    # find the pixel val that contains a specific amount of data below it
    def m1_weighted_percentile(self, hist, percentile):
        hist_percentile = np.sum(hist) * percentile / 100
        temp_sum = 0
        for i in range(np.size(hist)):
            temp_sum += hist[i]
            if temp_sum >= hist_percentile:
                return i
        return hist[-1]

    # method 2
    # finds the lowest and highest vals in the histogram that is greater than 
    #     a percentage of the peak val
    def m2_percent_distribution(self, hist):
        peak = np.amax(hist)
        peak_index = hist.tolist().index(peak)
        cutoff = self.m2_cutoff * peak
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
    def algorithm(self, c, d, img):
        a = self.algo_a; b = self.algo_b
        scalar = (b-a)/(d-c)

        algo_applied = np.add(img, - c)
        algo_applied = np.multiply(algo_applied, scalar)
        algo_applied = np.add(algo_applied, a)

        np.putmask(algo_applied, algo_applied > 255, 255)
        np.putmask(algo_applied, algo_applied < 0, 0)
        algo_applied = algo_applied.astype("uint8")

        return algo_applied

    
    # applies m1/m2 stretch and algorithm methods
    def apply_stretch(self, hist, img = None):
        if img is None:
            img = self.img
        
        c1 = self.m1_weighted_percentile(hist, self.m1_bounds)
        d1 = self.m1_weighted_percentile(hist, 100-self.m1_bounds)
        m1_img = self.algorithm(c1, d1, img)

        c2, d2 = self.m2_percent_distribution(hist)
        m2_img = self.algorithm(c2, d2, img)

        return m1_img, m2_img
    
    
    def hist_equalization_1(self, img = None):
        if img is None:
            img = self.img
        
        norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img1 = (255*norm_img1).astype(np.uint8)
        return norm_img1

    
    def hist_equalization_2(self, img = None):
        if img is None:
            img = self.img
        
        norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img2 = np.clip(norm_img2, 0, 1)
        norm_img2 = (255*norm_img2).astype(np.uint8)
        return norm_img2
    
    def display(self):
        # if the image is 3 channel, display the 3 channel versions
        if len(self.img.shape) == 3:
            self.rgb_display(self.img, self.hists['r'], self.hists['g'], self.hists['b'], 'Orig')
            self.rgb_display(self.new_img['stretch_m1'], self.hists['r_s1'], self.hists['g_s1'], self.hists['b_s1'], 'Stretched M1')
            self.rgb_display(self.new_img['stretch_m2'], self.hists['r_s2'], self.hists['g_s2'], self.hists['b_s2'], 'Stretched M2')
            self.rgb_display(self.new_img['equalization_m1'], self.hists['r_e1'], self.hists['g_e1'], self.hists['b_e1'], 'Equalized M1')
            self.rgb_display(self.new_img['equalization_m2'], self.hists['r_e2'], self.hists['g_e2'], self.hists['b_e2'], 'Equalized M2')
        # if the img is bw display the bw versions
        else:
            self.bw_display(self.img, self.hists['bw'], 'Orig')
            self.bw_display(self.new_img['stretch_m1'], self.hists['bw_s1'], 'Stretched M1')
            self.bw_display(self.new_img['stretch_m2'], self.hists['bw_s2'], 'Stretched M2')
            self.bw_display(self.new_img['equalization_m1'], self.hists['bw_e1'], 'Equalized M1')
            self.bw_display(self.new_img['equalization_m2'], self.hists['bw_e2'], 'Equalized M2')
            
    
    def rgb_display(self, img, r_hist, g_hist, b_hist, imgType):
        fig, ax = plt.subplots(2, 2, figsize=[24,16])
        
        ax[0,0].imshow(img)
        ax[0,0].set_title(f'{imgType} Image')
        ax[0,0].axis('off')

        ax[0,1].plot(r_hist, color='red')
        ax[0,1].set_title(f'Red {imgType} Hist')

        ax[1,0].plot(g_hist, color='green')
        ax[1,0].set_title(f'Green {imgType} Hist')

        ax[1,1].plot(b_hist, color='blue')
        ax[1,1].set_title(f'Blue {imgType} Hist')

        plt.show()
    
    def bw_display(self, img, hist, imgType):
        fig, ax = plt.subplots(1, 2, figsize=[24,8])
        plt.rcParams['image.cmap'] = 'gray'
        ax[0].imshow(img)
        ax[0].set_title(f'{imgType} Image')
        ax[0].axis('off')
        
        ax[1].plot(hist)
        ax[1].set_title(f'Grey {imgType} Hist')
        
        plt.show()