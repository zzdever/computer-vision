import numpy as np
import cv2
import random


# gaussian distribution    
Gaussian = np.array([[ 0.07511361,  0.1238414 ,  0.07511361],
       [ 0.1238414 ,  0.20417996,  0.1238414 ],
       [ 0.07511361,  0.1238414 ,  0.07511361]])



class MeanShift():
    def __init__(self, bandwidth_s=10, bandwidth_r=8):
        self._point_set = point_set
        #print self._point_set
        self._bandwidth_s = bandwidth_s
        self._bandwidth_r = bandwidth_r
        self._threshold = 1
        self._maxIteration = 100
        
    '''
    def getSeeds(self):
        # Bin points
        bin_sizes = defaultdict(int)
        for point in self._point_set:
            binned_point = np.cast[np.int32](point / bin_size)
            bin_sizes[tuple(binned_point)] += 1
        # Select only those bins as seeds which have enough members
        bin_seeds = np.array([point for point, freq in six.iteritems(bin_sizes) if 
                                freq >= min_bin_freq], dtype=np.float32)
        bin_seeds = bin_seeds * bin_size
        
        return bin_seeds
    '''
        
        
        
    def findPointsInWindow(self, point):
        points = []
        for p in self._point_set:
            if self.lenOfVector(p[:2]-point[:2])<self._bandwidth_s and self.lenOfVector(p[2:]-point[2:])<self._bandwidth_r:
                points.append(p)
                
        return points
        
        
    # Epanechnikov kernel
    # phi(x) = 3/4 * (1 - x^2)      if abs(x)<=1
    # phi(x) = 0                    else
    # K = C/(hs^2 * hr^3) *k()*k()
    def kernelEpanechnikov(self, x):
        if abs(x) <= 1:
            return (1-x*x)*3/4
        else:
            return 0
        
    def dkernelEpanechnikov(self, x):
        if abs(x) <= 1:
            return -x*3/2
        else:
            return 0
        
    def weightedMean(self, centroid, points_in_window):
        numerator = np.zeros(len(centroid))
        denominator = 0.0
        for point in points_in_window:
            # can be replaced by other kernel functions
            s_norm = self.lenOfVector( (point[:2] - centroid[:2])/self._bandwidth_s )
            r_norm = self.lenOfVector( (point[2:] - centroid[2:])/self._bandwidth_r )
            g = self.dkernelEpanechnikov(s_norm)*self.kernelEpanechnikov(r_norm) + self.kernelEpanechnikov(s_norm)*self.dkernelEpanechnikov(r_norm)
            numerator = numerator + point*g
            denominator = denominator + g
        
        print numerator
        print denominator
        return numerator / denominator
        
        
    def lenOfVector(self, vector):
        return sum([i*i for i in vector])
            
    def meanShift(self):
        shifted = []
        for point in self._point_set:
            centroid = point
            iteration = 0
    
            while True:
                points_in_window = self.findPointsInWindow(centroid)
                if len(points_in_window) == 0:
                    break
                
                weighted_mean = self.weightedMean(centroid, points_in_window)
                mean_shift = weighted_mean - centroid
                if self.lenOfVector(mean_shift) < self._threshold or iteration >= self._maxIteration:
                    break
                else:
                    centroid = weighted_mean                
                    iteration = iteration + 1
            shifted.append([point[0], point[1], centroid[2], centroid[3], centroid[4]])
        
        return shifted
            
                

if __name__=='__main__':
    filename = '/Users/ying/Italy4.jpg'
    
    img_origin = cv2.imread(filename)
    img_seg = img_origin.copy()
    #img2 = np.zeros((len(img), len(img[0]),2), np.int32)    #used to store differential
    
    width = len(img_origin[0])
    height = len(img_origin)
    
    
    pointSet = []
    for x in range(width):
        for y in range(height):
            point = [x, y]
            point.extend([chanel for chanel in img_origin[y][x]])
            pointSet.append(np.cast[np.int32](point))
            
    mean_shifter = MeanShift()
    print 'ss ms'
    shifted = mean_shifter.meanShift(img_origin)
            


    cv2.namedWindow('image')
    cv2.imshow('image', img_origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    