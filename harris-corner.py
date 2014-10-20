# author: ying

import cv2
import numpy as np
import random

WINDOW_SIZE = 5

filename = 'pic.jpg'
imgori = cv2.imread(filename)
img = cv2.imread(filename, 0)   # import as grey value
img2 = np.zeros((len(img), len(img[0]),2), np.int32)    #used to store differential
img3 = np.zeros((len(img), len(img[0])), np.int32)  #used to store R

# gaussian distribution    
Gaussian = np.array([[ 0.07511361,  0.1238414 ,  0.07511361],
       [ 0.1238414 ,  0.20417996,  0.1238414 ],
       [ 0.07511361,  0.1238414 ,  0.07511361]])

# compute differential for every pixel            
def Differential():
    for x in range(1, len(img[0])-1):
        for y in range(1, len(img)-1):
            # computer two differential
            dX = (int(img[y-1][x+1])-int(img[y-1][x-1])) + 2*(int(img[y][x+1])-int(img[y][x-1])) + (int(img[y+1][x+1])-int(img[y+1][x-1]))
            dY = (int(img[y+1][x+1])-int(img[y-1][x+1])) + 2*(int(img[y+1][x])-int(img[y-1][x])) + (int(img[y+1][x-1])-int(img[y-1][x-1]))
            img2[y][x] = [dX,dY]

# compute R
def RScore():
    for y in range(WINDOW_SIZE/2, len(img)-WINDOW_SIZE/2):
        for x in range(WINDOW_SIZE/2, len(img[0])-WINDOW_SIZE/2):
            M = np.zeros((2,2), np.int32)
            # computer the sum of differential products
            for u,v in [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]:
                dX,dY = int(img2[y+v][x+u][0]), int(img2[y+v][x+u][1])  # get dX,dY
                m = np.array([[dX*dX,dX*dY],
                      [dX*dY,dY*dY]], dtype=np.int32)
                M = M + m*Gaussian[v+1][u+1]    # sum of matrix
            # compute R
            k = 0.01    # set according to experience
            det = np.linalg.det(M)  # det of M
            trace = np.trace(M) # trace of M
            R = det - k*trace*trace
            img3[y][x] = R
            

def Find():
    threshold = np.max(img3)/100    # set according to the maximum value
    for y in range(WINDOW_SIZE/2, len(img)-WINDOW_SIZE/2):
        for x in range(WINDOW_SIZE/2, len(img[0])-WINDOW_SIZE/2):
            # ignore pixels whose R is small
            if img3[y][x] < threshold:
                continue
            else:
                flag = 0
                # get the largest, avoid duplicates
                for u,v in [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]:
                    if img3[y+v][x+u] >= img3[y][x]:
                        flag = 1
                        break
                if flag==0:
                    # draw a circle
                    cv2.circle(imgori,(x,y),7,(0,255,255),1,8,0) #point, radius, color, line size, 
                
        
        
Differential()
RScore()
Find()

cv2.namedWindow('image')
cv2.imshow('image', imgori)
cv2.waitKey(0)
cv2.destroyAllWindows()


