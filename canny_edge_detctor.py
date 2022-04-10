from scipy import ndimage
from scipy.ndimage import convolve
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm
import cv2
import sys

def rgb2gray(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_image

def load_images(dir_name = 'imgs'):    
    images_list = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            image_ready = mpimg.imread(dir_name + '/' + filename)
            image_ready = rgb2gray(image_ready)
            images_list.append(image_ready)
    return images_list

def visualize(images, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, image in enumerate(images):
        if image.shape[0] == 3:
            image = image.transpose(1,2,0)
        plt_index = i+1
        plt.subplot(2, 2, plt_index)
        plt.imshow(image, format)
    plt.show()

class cannyEdgeDetector:
    def __init__(self, images, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, low_threshold=0.05, high_threshold=0.15):
        self.images = images
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.low_Threshold = low_threshold
        self.high_Threshold = high_threshold
        self.images_final = []
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        return

    # appliying gaussian kernel(5) to smoothen the image
    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        sigma_constant = (2.0*sigma**2)
        constant = 1/(np.pi * sigma_constant)
        guass = np.exp(-((x**2 + y**2) / sigma_constant)) * constant
        return guass

    # getting magnitude and slope of the gradient here
    def sobel_filters(self, image):
        Kernal_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Kernal_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = ndimage.convolve(image, Kernal_x)
        Iy = ndimage.convolve(image, Kernal_y)
        magnitude = np.sqrt(Ix**2 + Iy ** 2)
        magnitude = np.absolute(magnitude)
        slope = np.arctan2(Iy, Ix)
        return (magnitude, slope)
    
    def non_max_suppression(self, image, slope):
        X, Y = image.shape
        Z = np.zeros((X,Y), dtype=np.int32)
        angle = slope * 180./np.pi
        angle[angle < 0] += 180
        #comparing with the surrounding pixel intensities
        for i in range(1,X-1):
            for j in range(1,Y-1):
                try:
                    Intense_x = 255
                    Intense_y = 255
                   #at angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        Intense_x = image[i, j+1]
                        Intense_y = image[i, j-1]
                    #at angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        Intense_x = image[i+1, j-1]
                        Intense_y = image[i-1, j+1]
                    #at angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        Intense_x = image[i+1, j]
                        Intense_y = image[i-1, j]
                    #at angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        Intense_x = image[i-1, j-1]
                        Intense_y = image[i+1, j+1]
                    if (image[i,j] >= Intense_x) and (image[i,j] >= Intense_y):
                        Z[i,j] = image[i,j]
                    else:
                        Z[i,j] = 0
                except IndexError as e:
                    pass
        return Z
    #getting final edge based on strong and weak piels
    def threshold(self, image):

        high_Threshold = image.max() * self.high_Threshold
        low_Threshold = high_Threshold * self.low_Threshold
        X, Y = image.shape
        img = np.zeros((X,Y), dtype=np.int32)
        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)
        strong_i, strong_j = np.where(image >= high_Threshold)
        zeros_i, zeros_j = np.where(image < low_Threshold)
        weak_i, weak_j = np.where((image <= high_Threshold) & (image >= low_Threshold))
        img[strong_i, strong_j] = strong
        img[weak_i, weak_j] = weak
        return (img)
    
    #finding if atleast one of the surround pixel is strong
    def hysteresis(self, image):

        X, Y = image.shape
        weak = self.weak_pixel
        strong = self.strong_pixel
        for i in range(1, X-1):
            for j in range(1, Y-1):
                if (image[i,j] == weak):
                    try:
                        if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                            or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                            or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                            image[i, j] = strong
                        else:
                            image[i, j] = 0
                    except IndexError as e:
                        pass
        return image
    
    def detect(self):
        images_final = []
        for i, image in enumerate(self.images):    
            self.image_smoothed = convolve(image, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.magnitude, self.slope = self.sobel_filters(self.image_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.magnitude, self.slope)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            image_final = self.hysteresis(self.thresholdImg)
            self.images_final.append(image_final)
        return self.images_final
imgs = load_images()
dictionary = {}
visualize(imgs, 'gray')
detector = cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, low_threshold=0.09, high_threshold=0.17, weak_pixel=100)
images_final = detector.detect()
visualize(images_final, 'gray')
#print(type(imgs_final))
np.set_printoptions(threshold=sys.maxsize)
'''print(imgs_final)
for i in imgs_final:
    print(type(i))
for elem in np.nditer(imgs_final):
    if elem == 255:
        elem = elem / 255'''
count = 0
total_sum = 0
for index, elem in np.ndenumerate(images_final):
    total_sum+= 1
    if elem == 255:
        #print(index)
        count += 1
print(total_sum)
print(count)
    
