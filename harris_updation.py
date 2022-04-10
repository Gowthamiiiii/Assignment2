import cv2
import numpy as np
from matplotlib import pyplot as plt

def harris_corner(directory,window_size,k,threshold):
    img = cv2.imread(directory)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if img is None:
        print('Try again' + directory)
        return None
    else:    
        height = img.shape[0]   
        width = img.shape[1]    
        Z = np.zeros((height,width))

    # applying sobel filter
    def sobel_filters(image):
        Kernal_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Kernal_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix = ndimage.convolve(image, Kernal_x)
        Iy = ndimage.convolve(image, Kernal_y)
        magnitude = np.sqrt(Ix**2 + Iy ** 2)
        magnitude = np.absolute(magnitude)
        slope = np.arctan2(Iy, Ix)
        return (magnitude, slope)    
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    Ix2=np.square(Ix)
    Iy2=np.square(Iy)
    Ixy=Ix*Iy
    offset = int( window_size / 2 )
    print ("Finding Corners")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            x2 = np.sum(Ix2[y-offset:y+1+offset, x-offset:x+1+offset])
            y2 = np.sum(Iy2[y-offset:y+1+offset, x-offset:x+1+offset])
            xy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            H = np.array([[x2,xy],[xy,y2]])
            det=x2*y2 - xy ** 2   #determinent
            tr= x2 + y2    #trace
            R=det-k*(tr**2)
            Z[y-offset, x-offset]=R
    cv2.normalize(Z, Z, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            result=Z[y, x]
            if result>threshold:
                cv2.circle(img,(x,y),3,(0,0,255))
                print(x,y)
    plt.figure("Manual Harris detector")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Harris corner detector")
    plt.xticks([]), plt.yticks([])
    plt.show()
harris_corner("gowthamipic.png", 5, 0.05, 0.30) 
