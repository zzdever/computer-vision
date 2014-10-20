import cv2  
import numpy as np  
import random
import sys

K = 10
ITERATION = 100
CENTROID_THRES = 1
  
#read image  
img = cv2.imread('image.jpg',cv2.IMREAD_COLOR)  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
#cv2.imshow('origin',img);  
  
#SIFT  
detector = cv2.SIFT()  
keypoints,descriptor = detector.detectAndCompute(gray,None)


total = len(descriptor)

cluster = set([])
while True:
    cluster.add(random.randint(0,total-1))
    if len(cluster) >= K:
        break
        
cluster = [[i] for i in cluster]
centroid = [[np.zeros(128, np.float32)] for i in cluster]
centroidOld = [[np.zeros(128, np.float32)] for i in cluster]

for index in range(len(cluster)):
    summ = np.zeros(128, np.float32)
    for i in cluster[index]:
        summ = summ + descriptor[i]
    centroid[index] = summ / len(cluster[index])
    centroidOld[index] = centroid[index].copy()
    

for ite in range(ITERATION):
    #print centroid[0]
    for item in range(len(descriptor)):
        distance_min = cv2.norm(descriptor[item], centroid[0])
        min_index = 0
        for index in range(1, len(cluster)):
            distance = cv2.norm(descriptor[item], centroid[index])
            if distance < distance_min:
                distance_min = distance
                min_index = index
        
        #print 'found cluster to put in:', min_index, 'for',item
        cluster[min_index].append(item)
        centroid[min_index] = (centroid[min_index]*(len(cluster[min_index])-1) + descriptor[item]) / len(cluster[min_index])
    
    
    #print centroid[0]
    
    summ = 0
    #print centroidOld[0] , centroid[0]
    for ii in range(len(centroid)):
        summ = summ + cv2.norm(centroid[ii], centroidOld[ii])
        
    print 'centroid distance after',ite, 'times:', summ
        
    if summ < CENTROID_THRES:
        print 'Converged after', ite, 'iterations'
        break
    else:
        for index in range(len(centroid)):
            centroidOld[index] = centroid[index].copy()
    

print 'iteration:',ite
print cluster

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
        
#img = cv2.drawKeypoints(gray,keypoints)  
#img2 = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
cv2.imshow('test',img)
#cv2.imshow('test2',img2); 
cv2.waitKey(0)  
cv2.destroyAllWindows()  
