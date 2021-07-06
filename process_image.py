import numpy as np
import cv2
import math
from contrast_stretch import apply_stretch_m1, apply_stretch_m2
from matplotlib import pyplot as plt
from IAGCWD import iagcwd

class ProcessImg:
    
    #TODO
    #    simplify init
    #    follow python naming convention for functions and class vars
    #    throw errors if wrong var types passed

    # assumes img is bgr
    def __init__(self, img, makeHist = True, color = True):
        self.img = {}
        self.hist = {}
        self.img['Orig'] = img
        self.isColor = color
        self.makeHist = makeHist
        
        if self.isColor:
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2HSV)
            self.hist['Orig'] = cv2.calcHist([self.img['Orig'][:,:,2]], [0], None, [256], [0,256]) 
        else:
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2GRAY)
            self.hist['Orig'] = cv2.calcHist([self.img['Orig']], [0], None, [256], [0,256])
    
    
    def enhancement_name(self, algorithm, src = 'Orig'):
        if src == 'Orig':
            return algorithm
        else:
            return src + " to " + algorithm
    
    # method in progress, to simplify code
#     def enhance(self, algorithm, func, src = 'Orig'):
#         hist = self.hist[src]
#         img = self.img[src]
#         # creates name for dictionaries
#         name = self.enhancement_name(algorithm, src)
        
#         # contrast stretch requires the hist in addition to the img as a parameter
#         if 'stretch' in algorithm:
#             if self.isColor:
#                 v = func(hist, img[:,:,2])
#                 self.img[name] = cv2.merge((img[:,:,0], img[:,:,1], v))
#             else:
#                 self.img[name] = func(hist, img)
        
#         else:
#             if self.isColor:
#                 v = func(img[:,:,2])
#                 self.img[name] = cv2.merge((img[:,:,0], img[:,:,1], v))
#             else:
#                 self.img[name] = func(img)
#         self.make_hist(name)
    
    
    def apply_contrast_stretch_m1(self, src = "Orig"):
        hist = self.hist[src]
        img = self.img[src]
        name = self.enhancement_name("stretch_m1", src)
        if self.isColor:
            stretched_v1 = apply_stretch_m1(hist, img[:,:,2])
            self.img[name] = cv2.merge((img[:,:,0], img[:,:,1], stretched_v1))
        else:
            self.img[name] = apply_stretch_m1(hist, img)
        self.make_hist(name)
    
    
    def apply_contrast_stretch_m2(self, src = "Orig"):
        hist = self.hist[src]
        img = self.img[src]
        name = self.enhancement_name("stretch_m2", src)
        if self.isColor:
            stretched_v2 = apply_stretch_m2(hist, img[:,:,2])
            self.img[name] = cv2.merge((img[:,:,0], img[:,:,1], stretched_v2))
        else:
            self.img[name] = apply_stretch_m2(hist, img)
        self.make_hist(name)
    
    
    def apply_equalization(self, src = "Orig"):
        img = self.img[src]
        name = self.enhancement_name("equalize", src)
        if self.isColor:
            equalized_v = cv2.equalizeHist(img[:,:,2])
            self.img[name] = cv2.merge((img[:,:,0], img[:,:,1], equalized_v))
        else:
            self.img[name] = cv2.equalizeHist(img)
        self.make_hist(name)
    
    
    def apply_iagcwd(self, src = "Orig"):
        img = self.img[src]
        name = self.enhancement_name("iagcwd", src)
        if self.isColor:
            iagcwd_v = iagcwd(img[:,:,2])
            self.img[name] = cv2.merge((img[:,:,0], img[:,:,1], iagcwd_v))
        else:
            self.img[name] = iagcwd(img)
        self.make_hist(name)
    
    
    def sigmoid_stretch(self, v_in, cutoff = 0.0, gain = 1.0, num_tones = 256):
        v_min = np.amin(v_in)
        v_range = np.amax(v_in) - v_min
        a = 1 / (1 + math.exp(gain * cutoff) )
        b = 1 / (1 + math.exp(gain * (cutoff - 1))) - a
        
        v = (v_in - v_min) / v_range
        v_out = (1 / (1 + np.exp(gain * (cutoff - v))) - a ) / b * num_tones
        np.putmask(v_out, v_out > 255, 255)
        np.putmask(v_out, v_out < 0, 0)
        v_out = v_out.astype(np.uint8)
        return v_out
    
    
    def apply_sigmoid_stretch(self, src = 'Orig'):
        img = self.img[src]
        name = self.enhancement_name("sigmoid", src)
        if self.isColor:
            sigmoid_v = self.sigmoid_stretch(img[:,:,2])
            self.img[name] = cv2.merge((img[:,:,0], img[:,:,1], sigmoid_v))
        else:
            self.img[name] = self.sigmoid_stretch(img)
        self.make_hist(name)
        
    
    # improve the implementation
    def display(self, cols = 2, height = 240):
        length = len(self.img)
        imgs = []
        imgs_hori = []
        for key in self.img:
            imgs.append(self.display_helper(key, height))
        while (length%cols != 0):
            imgs.append(np.zeros(imgs[0].shape, np.uint8))
            length += 1
        for num in range((length+1)//cols):
            imgs_hori.append(cv2.hconcat([*imgs[cols*num:cols*num+cols]]))
        grid = cv2.vconcat(imgs_hori)
        cv2.imshow("Images", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
    
    # returns resized img with label, in BGR, assuming img is colored
    def display_helper(self, img_algo, height, loc = (50,50)):
        if self.isColor:
            img = cv2.cvtColor(self.img[img_algo], cv2.COLOR_HSV2BGR)
        else:
            img = self.img[img_algo].copy()
        dim = (int(height*img.shape[1]/img.shape[0]), height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = cv2.putText(img, img_algo, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        return img
        
    
    def plot(self):
        for key in self.hist:
            self.plot_helper(key)
    
    
    def plot_helper(self,img_algo):
        hist = self.hist[img_algo]
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
    
    
    def make_hist(self,img_algo):
        if img_algo in self.img and self.makeHist:
            if self.isColor:
                self.hist[img_algo] = cv2.calcHist([self.img[img_algo][:,:,2]], [0], None, [256], [0,256])
            else:
                self.hist[img_algo] = cv2.calcHist([self.img[img_algo]], [0], None, [256], [0,256])
                
    
    def show_diff(self, src1, src2 = 'Orig', display = True, print_avg = True):
        img1 = self.img[src1]
        img2 = self.img[src2]
        
        diff = np.absolute(img1[:,:,2].astype(np.int16) - img2[:,:,2].astype(np.int16))
        diff = diff.astype(np.uint8)
        avg_diff = np.average(diff)
        
        if print_avg:
            print(f'avg difference in brightness between {src1} and {src2} is: {avg_diff}')
        if display:
            diff_img = cv2.merge((img1[:,:,0], img1[:,:,1], diff))
            diff_img = cv2.cvtColor(diff_img, cv2.COLOR_HSV2BGR)
            cv2.imshow('diff', diff_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return avg_diff