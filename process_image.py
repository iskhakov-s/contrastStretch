import numpy as np
import cv2
from matplotlib import pyplot as plt

class ProcessImg:
    
    m1_bounds = 5
    m2_cutoff = 0.05
    algo_a = 0
    algo_b = 255
    hists = {}
    img = {}
    
    #TODO
    #    simplify init
    #    follow python naming convention
    
    # changes: rgb to hsv, equalization method changed from normalize to equalizehist
    
    def __init__(self, img, color = True):
        self.img['Orig'] = img
        self.isColor = color
        if self.isColor:
            # make the color hist, stretch value channel, combine and make hist for stretched
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(self.img['Orig'])
            self.hists['v'] = cv2.calcHist([v], [0], None, [256], [0,256])
            stretched_v1, stretched_v2 = self.apply_stretch(self.hists['v'], v)
            self.img['Stretched M1'] = cv2.merge((h, s, stretched_v1))
            self.img['Stretched M2'] = cv2.merge((h, s, stretched_v2))
            self.hists['v_s1'] = cv2.calcHist([stretched_v1], [0], None, [256], [0,256])
            self.hists['v_s2'] = cv2.calcHist([stretched_v2], [0], None, [256], [0,256])
            # equalize and make a new hist
            equalized_v = cv2.equalizeHist(v)
            self.hists['v_e'] = cv2.calcHist([equalized_v], [0], None, [256], [0,256])
            self.img['Equalized'] = cv2.merge((h, s, equalized_v))
        else:
            # make the grey hist, stretch it, and make a new hist
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2GRAY)
            self.hists['bw'] = cv2.calcHist([self.img['Orig']], [0], None, [256], [0,256])
            self.img['Stretched M1'], self.img['Stretched M2'] = self.apply_stretch(self.hists['bw'])
            self.hists['bw_s1'] = cv2.calcHist([self.img['Stretched M1']], [0], None, [256], [0,256])
            self.hists['bw_s2'] = cv2.calcHist([self.img['Stretched M2']], [0], None, [256], [0,256])
            # equalize and make a new hist
            self.img['Equalized'] = cv2.equalizeHist(self.img['Orig'])
            self.hists['bw_e'] = cv2.calcHist([self.img['Equalized']], [0], None, [256], [0,256])
    
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
            img = self.img['Orig']
        
        c1 = self.m1_weighted_percentile(hist, self.m1_bounds)
        d1 = self.m1_weighted_percentile(hist, 100-self.m1_bounds)
        m1_img = self.algorithm(c1, d1, img)

        c2, d2 = self.m2_percent_distribution(hist)
        m2_img = self.algorithm(c2, d2, img)

        return m1_img, m2_img
    
    
    def display(self):
        # if the image is 3 channel, display the value hists
        if len(self.img['Orig'].shape) == 3:
            self.display_helper(self.hists['v'], 'Orig')
            self.display_helper(self.hists['v_s1'], 'Stretched M1')
            self.display_helper(self.hists['v_s2'], 'Stretched M2')
            self.display_helper(self.hists['v_e'], 'Equalized')
        # if the img is bw display the bw hists
        else:
            self.display_helper(self.hists['bw'], 'Orig')
            self.display_helper(self.hists['bw_s1'], 'Stretched M1')
            self.display_helper(self.hists['bw_s2'], 'Stretched M2')
            self.display_helper(self.hists['bw_e'], 'Equalized')
            
    
    def display_helper(self, hist, img_algo):
        img = self.img[img_algo]
        fig, ax = plt.subplots(1, 2, figsize=[15,5])
        if self.isColor:
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            plt.rcParams['image.cmap'] = 'gray'
        
        ax[0].imshow(img)
        ax[0].set_title(f'{img_algo} Image')
        ax[0].axis('off')
        
        ax[1].plot(hist)
        ax[1].set_title(f'{img_algo} Hist')
        
        plt.show()