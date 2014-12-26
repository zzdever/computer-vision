import numpy as np
import cv2
import random
import types
import sys


class MeanShift():
    def __init__(self, img, bandwidth_s=5, bandwidth_r=4):
        self._bandwidth_s = bandwidth_s
        self._bandwidth_r = bandwidth_r
        self._max_iteration = 30
        
        self.img = img.copy()

        
    # calculate the length of a vector    
    def lenOfVector(self, vector):
        vector = np.cast[np.float32](vector)
        return sum([i*i for i in vector])
    
    # can be replaced by other kernel functions            
    # Epanechnikov kernel
    # phi(x) = 3/4 * (1 - x^2)      if abs(x)<=1
    # phi(x) = 0                    else
    # K = C/(hs^2 * hr^3) *k()*k()
    def kernelEpanechnikov(self, x):
        if abs(x) <= 1:
            return (1-x*x)*3.0/4
        else:
            return 0
            
    # derivative of the kernel function
    def dkernelEpanechnikov(self, x):
        if abs(x) <= 1:
            return -x*3.0/2
        else:
            return 0

    # points climb to the cluster peak
    def ascent(self, x, y):
        points_count = 0
        
        # numerator is a vector
        numerator = np.zeros(5, dtype=np.float32)
        # denominator is a scalar
        denominator = 0.0
        
        for nx in range(x-self._bandwidth_s, x+self._bandwidth_s+1):
            for ny in range(y-self._bandwidth_s, y+self._bandwidth_s+1):
                # self, skip
                if x==nx and y==ny:
                    continue
                # not in image, skip
                if nx<0 or ny<0 or nx>=len(self.img[0]) or ny>=len(self.img):
                    continue      
                
                # calculate ((x-xi)/h)^2
                # it's the product of spatial and color parts
                s_distance_norm = self.lenOfVector((np.array([nx,ny], dtype=np.float32) - np.array([x,y], dtype=np.float32)) / self._bandwidth_s)
                r_distance_norm = self.lenOfVector((np.cast[np.float32](self.img[ny][nx]) - np.cast[np.float32](self.img[y][x])) / self._bandwidth_r)
                
                # the point's weight is not zero
                if s_distance_norm < 1.0 and r_distance_norm < 1.0:
                    # calculate g
                    g = -(self.dkernelEpanechnikov(s_distance_norm)*self.kernelEpanechnikov(r_distance_norm) 
                        + self.kernelEpanechnikov(s_distance_norm)*self.dkernelEpanechnikov(r_distance_norm))     
                    
                    # add up numerator and denominator
                    numerator = numerator + g * np.array([nx,ny]+list(self.img[ny][nx]), dtype=np.float32)
                    denominator = denominator + g
                    
                    # count points, if zero, return error
                    points_count = points_count + 1
                    

                
        if denominator<0.0001 and points_count>0:
            print 'sum not right at',x,y
            
        # if zero, return error
        if points_count <= 0:
            return 0, 0
        else:
            return points_count, numerator/denominator
        
                
        
        
    def meanShift(self):
        width = len(self.img[0])
        height = len(self.img)

        # shifted is used to store new color
        shifted = np.zeros_like(self.img)
        # mark is used to mark if the point has been traced
        mark = np.zeros((height, width), dtype=np.uint8)

    
        for x in range(width):
            for y in range(height):
                # if traced, skip
                if mark[y][x]:
                    print y,x,'skipped'
                    continue
                    
                # use the point as initial centroid
                centroid = np.array([x, y] + list(self.img[y][x]), dtype=np.int16)
                iteration = 0
    
                # path is used to trace
                # the points in the path climb to the same peak
                # so they won't be shifted again
                path = set([])
                path.add((centroid[0], centroid[1]))
                while True:
                    # ascent once
                    points_in_window_count, weighted_mean = self.ascent(centroid[0], centroid[1])
                    if points_in_window_count < 1:
                        break
                    
                    # because in an image, the spatial part is discrete, it should be rounded
                    weighted_mean = np.cast[np.int16](np.round(weighted_mean))
                    # calculate the mean shift
                    mean_shift = weighted_mean - np.cast[np.float32](centroid)
                    # mean as new centroid
                    centroid = weighted_mean  
                    # update path
                    path.add((centroid[0], centroid[1]))
                    # if shift is small enough or iteration is big enough, quit
                    if self.lenOfVector(mean_shift) < 0.01*(self._bandwidth_s+self._bandwidth_r) or iteration >= self._max_iteration:
                        break
                    else:     
                        iteration = iteration + 1
                
                # assign new color
                for point in path:
                    shifted[point[1]][point[0]] = np.cast[np.uint8]([sum(centroid[2:])/3 for i in range(3)])
                    mark[point[1]][point[0]] = 1
                        
        return shifted
        
      
                

if __name__=='__main__':
    if len(sys.argv) < 2:
        print 'Usage: python mean-shift.py [pic filename]'
        sys.exit(0)
    else:
        filename = str(sys.argv[1])
            
    img_origin = cv2.imread(filename)
    if img_origin == None:
        print 'file open failed'
        sys.exit(0)
    
    
    img_seg = img_origin.copy()
    
    width = len(img_origin[0])
    height = len(img_origin)
    
    # do mean shift        
    mean_shifter = MeanShift(img_origin)
    shifted = mean_shifter.meanShift()
            

    cv2.namedWindow('image')
    cv2.imshow('image', shifted)
    cv2.imwrite('shifted.jpg', shifted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    