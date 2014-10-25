import cv2  
import numpy as np  
import random
import sys

K = 10
ITERATION = 100
CENTROID_THRES = 1
  
  
def Classify(centroid, descriptors, classMark):
    total = len(descriptors)

    centroidOld = [[np.zeros(128, np.float32)] for i in cluster]


    cluster = []
    for i in range(K):
        cluster.append([i])
        clusterMark[i] = i
        
    for index in range(len(cluster)):
        summ = np.zeros(128, np.float32)
        for i in cluster[index]:
            summ = summ + descriptor[i]
        centroid[index] = summ / len(cluster[index])
        centroidOld[index] = centroid[index].copy()
    

    for ite in range(ITERATION):
        for item in range(len(descriptor)):
            distance_min = cv2.norm(descriptor[item], centroid[0])
            min_index = 0
            for index in range(1, len(cluster)):
                distance = cv2.norm(descriptor[item], centroid[index])
                if distance < distance_min:
                    distance_min = distance
                    min_index = index
                
            if clusterMark[item] >= 0:
                cluster[clusterMark[item]].remove(item)
            cluster[min_index].append(item)
            clusterMark[item] = min_index
            centroid[min_index] = (centroid[min_index]*(len(cluster[min_index])-1) + descriptor[item]) / len(cluster[min_index])
        
        summ = 0
        for ii in range(len(centroid)):
            summ = summ + cv2.norm(centroid[ii], centroidOld[ii])
                
        if summ < CENTROID_THRES:
            print 'Converged after', ite, 'iterations'
            break
        else:
            for index in range(len(centroid)):
                centroidOld[index] = centroid[index].copy()
    
                
                
                
    
filename = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg']
  

descriptors = []
for index in range(2):#len(filename)):
    img = cv2.imread(filename[index],cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(filename[index],cv2.IMREAD_COLOR)  
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #SIFT  
    detector = cv2.SIFT()  
    keypoints,descriptor = detector.detectAndCompute(img,None)
    for des in descriptor:
        descriptors.append((index, des))



clusterMark = [-1 for i in range(len(descriptor))]

    
centroid = [[np.zeros(128, np.float32)] for i in cluster]

Classify(centroid, descriptors, classMark)



print 'iteration:',ite
j = 0
sum = 0
for i in cluster:
    print j,':',len(i)
    sum = sum + len(i)
    j = j + 1
print 'sum',sum

'''
rr = []
for i in range(len(centroid)):
    for j in range(len(centroid)):
        if i==j:
            continue
        rr.append(cv2.norm(centroid[i], centroid[j]))

rr.sort()
print rr
sys.exit(0)
'''
    
''''
color0 = 0
color1 = 0
color2 = 0
for clu in cluster:
    for item in clu:
        cv2.circle(img,(int(keypoints[item].pt[0]),int(keypoints[item].pt[1])),3,(color2,color1,color0),2,8,0) #point, radius, color, line size, 
    color0 = color0 + 120
    if color0>255:
        color0 = 0
        color1 = color1 + 120
    if color1>255:
        color1 = 0
        color2 = color2 + 120
'''
        
#img = cv2.drawKeypoints(gray,keypoints)  
#img2 = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
cv2.imshow('test',img)
#cv2.imshow('test2',img2); 
cv2.waitKey(0)  
cv2.destroyAllWindows()  
