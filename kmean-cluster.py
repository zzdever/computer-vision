import cv2  
import numpy as np  
import random
import sys

K = 10
ITERATION = 100
CENTROID_THRES = 1
  
'''  
centroid format
[c0,c1,c2,...,cN, sum average]
c0: [c00,c01,c02,...,c0N, sum average]


cluster format
[[indices],[],[],[],...,[]]
'''    
    
  
def Classify(cluster, descriptors):
    total = len(cluster)
    
    centroid = [[np.zeros(128, np.float32)] for i in range(K+1)]
    centroidOld = [[np.zeros(128, np.float32)] for i in range(K+1)]


    clusterN = []
    for i in range(K):
        clusterN.append([cluster[i]])
            
    clusterMark = {}
    for item in cluster:
        clusterMark[item] = -1
            
    for index in range(len(centroid)-1):
        centroid[index] = descriptors[clusterN[index][0]]['descriptor']
        centroidOld[index] = centroid[index].copy()

    for ite in range(ITERATION):
        for item in cluster:
            
            distance_min = cv2.norm(descriptors[item]['descriptor'], centroid[0])
            min_index = 0
            for index in range(1, len(centroid)-1):
                distance = cv2.norm(descriptors[item]['descriptor'], centroid[index])
                if distance < distance_min:
                    distance_min = distance
                    min_index = index
                
            if clusterMark[item] >= 0:
                clusterN[clusterMark[item]].remove(item)
            clusterN[min_index].append(item)
            clusterMark[item] = min_index
            centroid[min_index] = (centroid[min_index]*(len(clusterN[min_index])-1) + descriptors[item]['descriptor']) / len(clusterN[min_index])
        
        sumCentroid = 0
        sumDiff = 0
        for index in range(len(centroid)-1):
            sumCentroid = sumCentroid + centroid[index]
            sumDiff = sumDiff + cv2.norm(centroid[index], centroidOld[index])
                
        if sumDiff < CENTROID_THRES:
            print 'Converged after', ite, 'iterations'
            centroid[-1] = sumCentroid / (len(centroid)-1)
            break
        else:
            for index in range(len(centroid)-1):
                centroidOld[index] = centroid[index].copy()
                
                
    return centroid, clusterN
    
                

def FindClosest(all, query):
    isLeave = not isinstance(all[0],list)
      
    if isLeave:
        distance_min = cv2.norm(query, all[0])
    else:
        distance_min = cv2.norm(query, all[0][-1])        
        
    min_index = 0
    for index in range(1, len(all)):
        if isLeave:
            distance = cv2.norm(query, all[index])
        else:
            distance = cv2.norm(query, all[index][-1])
        if distance < distance_min:
            distance_min = distance
            min_index = index
            
    return min_index
    
    
               
                
####################Build tree##############################
filename = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg']

descriptors = []
for index in range(1):#len(filename)):
    img = cv2.imread(filename[index],cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(filename[index],cv2.IMREAD_COLOR)  
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #SIFT  
    detector = cv2.SIFT()  
    keypoints,descriptor = detector.detectAndCompute(img,None)
    for des in descriptor:
        descriptors.append({'descriptor':des, 'img':index})# 'cluster': (0,)})


cluster = [i for i in range(len(descriptors))]
centroid, cluster = Classify(cluster, descriptors) #return centroid,cluster
# build the second layer tree
for i in range(len(cluster)):
    centroid[i],cluster[i] = Classify(cluster[i], descriptors)



####################Query##############################

query = 'query.jpg'
imgq = cv2.imread(query,cv2.IMREAD_GRAYSCALE)

detector = cv2.SIFT()  
keypoints,descriptor = detector.detectAndCompute(img,None)

matchCount = [0 for i in filename]
for des in descriptor:
    min_index = []
    min_index.append(FindClosest(centroid[0:-1], des))
    min_index.append(FindClosest(centroid[min_index[0]][0:-1], des))
    
    for item in cluster[min_index[0]][min_index[1]]:
        imgNum = descriptors[item]['img']
        matchCount[imgNum] = matchCount[imgNum] + 1
    
print 'match count result:',matchCount
match = matchCount.index(max(matchCount))
img = cv2.imread(filename[match],cv2.IMREAD_GRAYSCALE)
    
cv2.imshow('query',imgq) 
cv2.imshow('match',img)

cv2.waitKey(0)  
cv2.destroyAllWindows()  

        
#img = cv2.drawKeypoints(gray,keypoints)  
#img2 = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
