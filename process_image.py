import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import exp

from contrast_stretch import apply_stretch_m1, apply_stretch_m2
from IAGCWD import iagcwd, agcwd
from sece import sece, sece_dct


def alpha_blend(img1, img2, alpha = 0.5):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    out = alpha * img1 + (1-alpha) *img2
    np.putmask(out, out > 255, 255)
    np.putmask(out, out < 0, 0)
    out = out.astype(np.uint8)
    return out


def apply_sigmoid_stretch(img, g = 10, c = .5):
    img = img.astype(np.float32)
    z = ( img - img.min() ) / (img.max() - img.min())
    a = 1 / (1 + exp(g * c))
    b = 1 / (1 + exp(g * (c-1))) - a
    out = 1 / (1 + np.exp(g * (c-z)))
    out = (out - a) / b * 255
    np.putmask(out, out > 255, 255)
    np.putmask(out, out < 0, 0)
    out = out.astype(np.uint8)
    return out


def apply_clahe(img, clip_limit=4.0, grid_size = (8,8)):
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = grid_size)
    return clahe.apply(img)


def novel_algorithm(img):
    """ New algorithm
    Splits image in two vertically, applies sigmoid stretch to each half
    then merges the images and blends it with the original image
    applies clahe to the result, then agcwd to that result
    returns the alpha blend of the clahe and agcwd image results
    """
    r1 = apply_sigmoid_stretch(img[:,:img.shape[1]//2])
    r2 = apply_sigmoid_stretch(img[:,img.shape[1]//2:])
    sig = np.hstack((r1,r2))
    a_blend = alpha_blend(sig, img)
    clahe = apply_clahe(a_blend)
    hist = cv2.calcHist([clahe], [0], None, [256], [0,256])
    agc = agcwd(clahe, hist)
    out = alpha_blend(clahe, agc)
    return out
    


class ProcessImg:
    
    # TODO
    #    simplify init
    #    follow python naming convention for functions and class vars
    #    throw errors if wrong var types passed or if invalid name is accessed
    
    func = {
                    'stretch_m1' : apply_stretch_m1,
                    'stretch_m2' : apply_stretch_m2,
                    'equalize' : cv2.equalizeHist,
                    'iagcwd' : iagcwd,
                    'agcwd' : agcwd,
                    'sece' : sece,
                    'sece_dct' : sece_dct,
                    'sigmoid' : apply_sigmoid_stretch,
                    'clahe' : apply_clahe,
                    'novel' : novel_algorithm
    }
    
    
    # assumes img is bgr
    def __init__(self, img, makeHist = True, color = True):
        self.img = {}
        self.hist = {}
        self.isColor = color
        self.makeHist = makeHist
        
        if self.isColor:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self.h, self.s, self.img['Orig'] = cv2.split(img)            
        else:
            self.img['Orig'] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        self.hist['Orig'] = cv2.calcHist([self.img['Orig']], [0], None, [256], [0,256])
    
    
    def get_name(self, algorithm, src = 'Orig'):
        if src == 'Orig':
            return algorithm
        return src + " to " + algorithm

    
#     method in progress, need to simplify code
    def enhance(self, algorithm, src = 'Orig'):
        img = self.img[src]
        # creates name for dictionaries
        name = self.get_name(algorithm, src)
        
        # contrast stretch requires the hist in addition to the img as a parameter
        if 'stretch' in algorithm or algorithm == 'agcwd':
            hist = self.hist[src]
            self.img[name] = self.func[algorithm](img, hist)
        else:
            self.img[name] = self.func[algorithm](img)
        
        self.make_hist(name)
        return self.img[name]
    
    
    def enhance_all(self, src = 'Orig'):
        for key in self.func:
            self.enhance(key, src)
    
    
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
    
    
    # assumes hsv
    def get_img(self, img):
        return cv2.merge((self.h, self.s, self.img[img]))
    
    
#     def save_img(self, img):
        