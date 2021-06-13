import numpy as np
import cv2
from contrast_stretch import apply_stretch
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
        self.hists = {}
        self.img['Orig'] = img
        self.isColor = color
        self.makeHist = makeHist
        
        if self.isColor:
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2HSV)
            self.h, self.s, self.v = cv2.split(self.img['Orig'])
            self.hists['Orig'] = cv2.calcHist([self.v], [0], None, [256], [0,256]) 
        else:
            self.img['Orig'] = cv2.cvtColor(self.img['Orig'], cv2.COLOR_BGR2GRAY)
            self.hists['Orig'] = cv2.calcHist([self.img['Orig']], [0], None, [256], [0,256])

                

    def apply_contrast_stretch(self, img_type = "Orig"):
        hist = self.hists[img_type]
        img = self.img[img_type]
        if self.isColor:
            stretched_v1, stretched_v2 = apply_stretch(self.hists['Orig'], self.v)
            self.img['Stretched M1'] = cv2.merge((self.h, self.s, stretched_v1))
            self.img['Stretched M2'] = cv2.merge((self.h, self.s, stretched_v2))
        else:
            self.img['Stretched M1'], self.img['Stretched M2'] = apply_stretch(hist, img)
        self.make_hist('Stretched M1')
        self.make_hist('Stretched M2')
    
    
    def apply_equalization(self, img_type = "Orig"):
        img = self.img[img_type]
        if self.isColor:
            equalized_v = cv2.equalizeHist(self.v)
            self.img['Equalized'] = cv2.merge((self.h, self.s, equalized_v))
        else:
            self.img['Equalized'] = cv2.equalizeHist(self.img['Orig'])
        self.make_hist('Equalized')
    
    
    def apply_iagcwd(self, img_type = "Orig"):
        img = self.img[img_type]
        if self.isColor:
            iagcwd_v = iagcwd(self.v)
            self.img['IAGCWD'] = cv2.merge((self.h, self.s, iagcwd_v))
        else:
            self.img['IAGCWD'] = iagcwd(self.img['Orig'])
        self.make_hist('IAGCWD')
    
    
    # improve the implementation
    def display(self):
        length = len(self.img)
        imgs = []
        imgs_vert = []
        for key in self.img:
            imgs.append(self.display_helper(key))
        for num in range((length+1)//2):
            if (num == length//2 and length%2 == 1):
                imgs_vert.append(cv2.vconcat([imgs[2*num], np.zeros(imgs[2*num].shape, np.uint8)]))
            else:
                imgs_vert.append(cv2.vconcat([imgs[2*num],imgs[2*num+1]]))
        grid = cv2.hconcat(imgs_vert)
        cv2.imshow("Images", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
    
    # returns resized img with label, in BGR, assuming img is colored
    def display_helper(self, img_algo, height = 360, loc = (50,50)):
        if self.isColor:
            img = cv2.cvtColor(self.img[img_algo], cv2.COLOR_HSV2BGR)
        else:
            img = self.img[img_algo].copy()
        dim = (int(height*img.shape[1]/img.shape[0]), height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = cv2.putText(img, img_algo, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        return img
        
    
    def plot(self):
        for key in self.hists:
            self.plot_helper(key)
    
    def plot_helper(self,img_algo):
        hist = self.hists[img_algo]
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
                self.hists[img_algo] = cv2.calcHist([self.img[img_algo][:,:,2]], [0], None, [256], [0,256])
            else:
                self.hists[img_algo] = cv2.calcHist([self.img[img_algo]], [0], None, [256], [0,256])
                