import numpy as np
import cv2
import random
import types


# gaussian distribution    
Gaussian = np.array([[ 0.07511361,  0.1238414 ,  0.07511361],
       [ 0.1238414 ,  0.20417996,  0.1238414 ],
       [ 0.07511361,  0.1238414 ,  0.07511361]])



class MeanShift():
    def __init__(self, img, bandwidth_s=5, bandwidth_r=4):
        #self._point_set = point_set
        #print self._point_set
        self._bandwidth_s = bandwidth_s
        self._bandwidth_r = bandwidth_r
        #self._threshold = 1
        self._max_iteration = 30
        
        self.img = img.copy()

        
        
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
        
    def dkernelEpanechnikov(self, x):
        if abs(x) <= 1:
            return -x*3.0/2
        else:
            return 0
        
    def weightedMean(self, centroid, points_in_window):
        numerator = np.zeros(len(centroid))
        denominator = 0.0
        for point in points_in_window:
            s_norm = self.lenOfVector( (np.cast[np.float32](point[:2]) - np.cast[np.float32](centroid[:2]))/self._bandwidth_s )
            r_norm = self.lenOfVector( (np.cast[np.float32](point[2:]) - np.cast[np.float32](centroid[2:]))/self._bandwidth_r )

            if abs(s_norm)>1 or abs(r_norm)>1:
                print 'not right norms',centroid,point,s_norm,r_norm
            g = -(self.dkernelEpanechnikov(s_norm)*self.kernelEpanechnikov(r_norm) + self.kernelEpanechnikov(s_norm)*self.dkernelEpanechnikov(r_norm))
            print 'g',g
            numerator = numerator + point*g
            denominator = denominator + g
        
        
        
        print 'numerator',numerator
        print 'denominator',denominator
        
        
        if abs(denominator) < 0.000001:
            ''''
            print centroid,s_norm,r_norm
            print 'numerator',numerator
            print 'denominator',denominator
            '''
            return 0
        
            
        return numerator / denominator

            
    
    
    def ascent(self, x, y):
        points_count = 0
        
        numerator = np.zeros(5, dtype=np.float32)
        denominator = 0.0
        
        for nx in range(x-self._bandwidth_s, x+self._bandwidth_s+1):
            for ny in range(y-self._bandwidth_s, y+self._bandwidth_s+1):
                if x==nx and y==ny:
                    continue
                if nx<0 or ny<0 or nx>=len(self.img[0]) or ny>=len(self.img):
                    continue      
                
                s_distance_norm = self.lenOfVector((np.array([nx,ny], dtype=np.float32) - np.array([x,y], dtype=np.float32)) / self._bandwidth_s)
                r_distance_norm = self.lenOfVector((np.cast[np.float32](self.img[ny][nx]) - np.cast[np.float32](self.img[y][x])) / self._bandwidth_r)
                
                if s_distance_norm < 1.0 and r_distance_norm < 1.0:
                    g = -(self.dkernelEpanechnikov(s_distance_norm)*self.kernelEpanechnikov(r_distance_norm) 
                        + self.kernelEpanechnikov(s_distance_norm)*self.dkernelEpanechnikov(r_distance_norm))
                        
                        
                    if abs(g) < 0.0001:
                        print 'not right',
                        print 'centroid',x,y,self.img[y][x],
                        print 'point',nx,ny,self.img[ny][nx]
                        print 's norm',s_distance_norm,
                        print 'r norm',r_distance_norm
                        
                    
                    numerator = numerator + g * np.array([nx,ny]+list(self.img[ny][nx]), dtype=np.float32)
                    denominator = denominator + g
                    
                    points_count = points_count + 1
                    

                
        if denominator<0.0001 and points_count>0:
            print 'sum not right at',x,y
            
        if points_count <= 0:
            return 0, 0
        else:
            #print 'numerator',numerator,
            #print 'denominator',denominator
            #print x,y,'at converge at',numerator/denominator
            return points_count, numerator/denominator
        
                
        
        
    def meanShift(self):
        width = len(self.img[0])
        height = len(self.img)

        shifted = np.zeros_like(self.img)
        mark = np.zeros((height, width), dtype=np.uint8)

    
        for x in range(width):
            for y in range(height):
                if mark[y][x]:
                    print y,x,'skipped'
                    continue
                    
                centroid = np.array([x, y] + list(self.img[y][x]), dtype=np.int16)
                iteration = 0
    
                #print 'doing', centroid
                path = set([])
                path.add((centroid[0], centroid[1]))
                while True:
                    points_in_window_count, weighted_mean = self.ascent(centroid[0], centroid[1])
                    if points_in_window_count < 1:
                        break
                    
                    #weighted_mean = np.cast[np.int16](np.round(self.weightedMean(centroid, points_in_window)))                    
                    weighted_mean = np.cast[np.int16](np.round(weighted_mean))
                    mean_shift = weighted_mean - np.cast[np.float32](centroid)


                    #print 'weighted_mean',weighted_mean
                    #print 'mean_shift', mean_shift
                    centroid = weighted_mean  
                    path.add((centroid[0], centroid[1]))
                    if self.lenOfVector(mean_shift) < 0.01*(self._bandwidth_s+self._bandwidth_r) or iteration >= self._max_iteration:
                        break
                    else:     
                        iteration = iteration + 1
                
                #print 'iteration',iteration
                for point in path:
                    shifted[point[1]][point[0]] = np.cast[np.uint8]([sum(centroid[2:])/3 for i in range(3)])
                    mark[point[1]][point[0]] = 1
                #print 'shifted',x,y,shifted[y][x]
                
            print x,'column done'
        
        return shifted
        
      
                

if __name__=='__main__':
    filename = '/Users/ying/tt.png'
    
    img_origin = cv2.imread(filename)
    img_seg = img_origin.copy()
    #img2 = np.zeros((len(img), len(img[0]),2), np.int32)    #used to store differential
    
    width = len(img_origin[0])
    height = len(img_origin)
    
            
    mean_shifter = MeanShift(img_origin)
    shifted = mean_shifter.meanShift()
            

    cv2.namedWindow('image')
    cv2.imshow('image', shifted)
    cv2.imwrite('/Users/ying/shifted.jpg', shifted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    