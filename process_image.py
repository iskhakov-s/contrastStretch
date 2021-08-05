import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import exp

from contrast_stretch import apply_stretch_m1, apply_stretch_m2
from IAGCWD import iagcwd
from sece import sece, sece_dct

class ProcessImg:
    
    #TODO
    #    simplify init
    #    follow python naming convention for functions and class vars
    #    throw errors if wrong var types passed or if invalid name is accessed

    # assumes img is bgr
    def __init__(self, img, makeHist = True, color = True):
        self.img = {}
        self.hist = {}
        self.img['Orig'] = img
        self.isColor = color
        self.makeHist = makeHist
        
        if self.isColor:
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2HSV)
            self.h, self.s, self.img['Orig'] = cv2.split(self.img['Orig'])            
        else:
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2GRAY)
        
        self.hist['Orig'] = cv2.calcHist([self.img['Orig']], [0], None, [256], [0,256])
    
    
    def get_name(self, algorithm, src = 'Orig'):
        if src == 'Orig':
            return algorithm
        return src + " to " + algorithm

    
#     method in progress, need to simplify code
#     def enhance(self, algorithm, src = 'Orig'):
#         hist = self.hist[src]
#         img = self.img[src]
#         # creates name for dictionaries
#         name = self.get_name(algorithm, src)
        
#         # contrast stretch requires the hist in addition to the img as a parameter
#         if 'stretch' in algorithm:
#             self.img[name] = func(hist, img)
        
#         else:
#             self.img[name] = func(img)
#         self.make_hist(name)
    
    
    def apply_contrast_stretch_m1(self, src = "Orig"):
        hist = self.hist[src]
        img = self.img[src]
        name = self.get_name("stretch_m1", src)
        self.img[name] = apply_stretch_m1(hist, img)
        self.make_hist(name)
        return self.img[name]
    
    
    def apply_contrast_stretch_m2(self, src = "Orig"):
        hist = self.hist[src]
        img = self.img[src]
        name = self.get_name("stretch_m2", src)
        self.img[name] = apply_stretch_m2(hist, img)
        self.make_hist(name)
        return self.img[name]
    
    
    def apply_equalization(self, src = "Orig"):
        img = self.img[src]
        name = self.get_name("equalize", src)
        self.img[name] = cv2.equalizeHist(img)
        self.make_hist(name)
        return self.img[name]
    
    
    def apply_iagcwd(self, src = "Orig"):
        img = self.img[src]
        name = self.get_name("iagcwd", src)
        self.img[name] = iagcwd(img)
        self.make_hist(name)
        return self.img[name]
    
    
    def apply_sece(self, src = "Orig"):
        img = self.img[src]
        name = self.get_name("sece", src)
        self.img[name] = sece(img)
        self.make_hist(name)
        return self.img[name]
    
    
    def apply_sece_dct(self, src = "Orig"):
        img = self.img[src]
        name = self.get_name("sece_dct", src)
        self.img[name] = sece_dct(img)
        self.make_hist(name)
        return self.img[name]
    
    
    # may not work bc imgs are int not float, may need conversion or rounding
    def sigmoid_stretch(self, v_in, a = 12, b = 6):
        c1 = 1 / (1 + exp(b) )
        c2 = 1 / (1 + exp(b-a)) - c1
        
        v_in = v_in.astype(np.float32)
        v_out = 1 / (1 + np.exp(-1/255 * (a*v_in - 255*b) ) )
        v_out = np.round( (v_out-c1)/c2 * 255 )
        
        np.putmask(v_out, v_out > 255, 255)
        np.putmask(v_out, v_out < 0, 0)
        v_out = v_out.astype(np.uint8)
        
        return v_out
    
    
    def apply_sigmoid_stretch(self, src = 'Orig'):
        img = self.img[src]
        name = self.get_name("sigmoid", src)
        self.img[name] = self.sigmoid_stretch(img)
        self.make_hist(name)
        return self.img[name]
    
    
#     def sigmoid_stretch(self, v_in, a = 1.6):
#         v_out = 255 / (1 + np.exp( -a * ( v_in - 127/32 ) ) )
#         np.putmask(v_out, v_out > 255, 255)
#         np.putmask(v_out, v_out < 0, 0)
#         v_out = v_out.astype(np.uint8)
#         return v_out
    
    
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
        return grid
                
    
    # returns resized img with label, in BGR, assuming img is colored
    def display_helper(self, img_algo, height, loc = (10,30)):
        if self.isColor:
            img = cv2.merge((self.h, self.s, self.img[img_algo]))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            img = self.img[img_algo].copy()
        dim = (int(height*img.shape[1]/img.shape[0]), height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = cv2.putText(img, img_algo, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        return img
        
    
    def plot(self):
        for key in self.hist:
            self.plot_helper(key)
        pass
    
    
    def plot_helper(self,img_algo):
        hist = self.hist[img_algo]
        img = self.img[img_algo]
        fig, ax = plt.subplots(1, 2, figsize=[15,5])
        if self.isColor:
            img = cv2.merge((self.h, self.s, img))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            plt.rcParams['image.cmap'] = 'gray'
        
        ax[0].imshow(img)
        ax[0].set_title(f'{img_algo} Image')
        ax[0].axis('off')
        
        ax[1].plot(hist)
        ax[1].set_title(f'{img_algo} Hist')
        
        plt.show()
        pass
    
    
    def make_hist(self,img_algo):
        if img_algo in self.img and self.makeHist:
            self.hist[img_algo] = cv2.calcHist([self.img[img_algo]], [0], None, [256], [0,256])
            return self.hist[img_algo]
        pass
                
    
    # assumes image is color and that both src are valid
    def show_diff(self, src1, src2 = 'Orig', display = True, print_avg = True):
        img1 = self.img[src1]
        img2 = self.img[src2]
        
        diff = np.absolute(img1.astype(np.int16) - img2.astype(np.int16))
        diff = diff.astype(np.uint8)
        avg_diff = np.average(diff)
        
        if print_avg:
            print(f'avg difference in brightness between {src1} and {src2} is: {avg_diff}')
        if display:
            diff_img = cv2.merge((self.h, self.s, diff))
            diff_img = cv2.cvtColor(diff_img, cv2.COLOR_HSV2BGR)
            cv2.imshow('diff', diff_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return avg_diff
    
    
    def get_img(self, img):
        return cv2.merge((self.h, self.s, self.img[img]))
    
    
#     def save_img(self, img):
        