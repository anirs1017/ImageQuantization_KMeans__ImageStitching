# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 03:46:01 2018

@author: Aniruddha Sinha
UB Person Number = 50289428
UBIT = asinha6@buffalo.edu
"""

UBIT = '<asinha6>'; import numpy as np; np.random.seed(sum([ord(c) for c in UBIT]))

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import time

'function to write the image to disk'
def writeImage(img, imageName):
    cv2.imwrite("results/" + imageName + ".jpg", img)

''' The following functions for drawing the epipolar lines and epipolar points and finding epilines
 have been referenced and partly been copied from https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html'''

def drawEpipolarLines(image1, image2, epilines, epipts1, epipts2, num):
    h, w = image1.shape
    
    colors = [[100, 200, 300], [1,3,2], [32, 43, 12], [90, 200, 12], [100, 100, 100], [0, 100, 200], [100, 0, 200], [100, 200, 0], [90, 30, 180], [120, 140, 150]]
    
    for i, r, point1, point2 in zip(range(10), epilines, epipts1, epipts2):
        color = tuple(colors[i])
        x0, y0 = map(int, [0, -r[2]/r[1]] )
        x1, y1 = map(int, [w, -(r[2] + r[0]*w)/r[1]] )
        
        global img1
        global img2
        
        if num == 1:
            image1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
            image1 = cv2.circle(img1, tuple(point1), 5, color, -1)
            image2 = cv2.circle(img2, tuple(point2), 5, color, -1)
        else:
            image1 = cv2.line(img2, (x0,y0), (x1,y1), color, 1)
            image1 = cv2.circle(img2, tuple(point1), 5, color, -1)
            image2 = cv2.circle(img1, tuple(point2), 5, color, -1)
    return image1
    
def findEpilines(left_img, right_img, pts1, pts2, F):
    
    #Find epilines corresponding to points in the right image and
    #draw their lines on left image
    lines_left = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines_left = lines_left.reshape(-1,3)
    epiLeftImg = drawEpipolarLines(left_img, right_img, lines_left, pts1, pts2, 1)

    
    #Find epilines corresponding to points in the left image and
    #draw their lines on right image
    lines_right = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines_right = lines_right.reshape(-1,3)
    epiRightImg = drawEpipolarLines(right_img,left_img, lines_right, pts2, pts1, 2)
       
    return epiRightImg, epiLeftImg


#################### Start Main ########################
'Start function execution'
t= time.time()

'Read the two images, first in colour and then in grayscales'
img1 = cv2.imread("data/tsucuba_left.png")
img2 = cv2.imread("data/tsucuba_right.png")            

tsucuba_left = cv2.imread("data/tsucuba_left.png", 0)
tsucuba_right = cv2.imread("data/tsucuba_right.png", 0)

############################# PART 1 ##################################################
print('\n\n##################### Starting execution of Task 2 part 1 ##########################################')

''' The following code for computing SIFT features and drawing keypoints has been referenced and partly copied from '''
''''https://docs.opencv.org/3.4.3/da/df5/tutorial_py_sift_intro.html'''''

'Compute the keypoints and features of the two images'
sift = cv2.xfeatures2d.SIFT_create()
keypoints1 = sift.detect(tsucuba_left, None)
keypoints2 = sift.detect(tsucuba_right, None)

'Draw the keypoints on the images and write to disk'
task2_sift1 = cv2.drawKeypoints(img1, keypoints1, None ,(0,0,255)) 
task2_sift2 = cv2.drawKeypoints(img2, keypoints2, None, (0,0,255))
writeImage(task2_sift1, "task2_sift1")
print('\n\n******************** Image task2_sift_1 successfully written to drive **************************')
writeImage(task2_sift2, "task2_Sift2")
print('\n\n******************** Image task2_sift_2 successfully written to drive **************************')


''' The following code for computing the K Nearest Neighbours and drawing matches
 has been referenced and party copied from - 
 https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html '''


'Find the keypoints and the descriptors of the keypoints'      
keypoint_left_img, descriptor_left_img = sift.detectAndCompute(tsucuba_left, None)
keypoint_rt_img, descriptor_rt_img = sift.detectAndCompute(tsucuba_right, None)

'Using FLANN matching technique for finding K Nearest Neighbours'
FLANN_INDEX_KDTREE = 0
check_index_params = dict(algorithm = FLANN_INDEX_KDTREE, tress = 5)
check_search_params = dict(checks=100)

'''Create object for FLANN matching'''
doFlann = cv2.FlannBasedMatcher(check_index_params, check_search_params)
NearestNeighborsMatches = doFlann.knnMatch(descriptor_left_img, descriptor_rt_img, k=2)

'Calculate the better neighbour of the two best matches of a keypoint of left image in the right image'
keepKeypoints = []
tsc_pts1 = []
tsc_pts2 = []
for m,n in NearestNeighborsMatches:
    if m.distance < 0.75*n.distance:
        keepKeypoints.append(m)
        tsc_pts2.append(keypoint_left_img[m.trainIdx].pt)
        tsc_pts1.append(keypoint_rt_img[m.queryIdx].pt)

'Draw the matched keypoints of the two images'
task2_matches_knn = cv2.drawMatches(img1, keypoint_left_img, img2, keypoint_rt_img, keepKeypoints, None, (0,255,0), (255,0,0) ,flags=2)
writeImage(task2_matches_knn, "task2_matches_knn")
print('\n\n******************** Image task2_matches_knn successfully written to drive **************************')
print('\n\n############### Task 2 part 1 successfully completed.#####################\nTime taken for Task 2 part 1 = ',time.time()-t,' seconds')
############################# END PART 1 ################################################


########################## PART 2 ###################################################
print('\n\n##################### Starting execution of Task 2 part 2 ##########################################')

''' The following code for finding Fundamental Matrix and computing the inliers has been referenced and 
party been copied from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html'''

MIN_NUM_MATCHES = 10

if len(keepKeypoints) > MIN_NUM_MATCHES:
    
    coord_left = np.array([ keypoint_left_img[m.queryIdx].pt for m in keepKeypoints ])
    np.reshape(coord_left, (-1,1,2))
    coord_left = np.int32(coord_left)
    coord_rt = np.array([ keypoint_rt_img[m.trainIdx].pt for m in keepKeypoints ])
    np.reshape(coord_rt, (-1,1,2))
    coord_rt = np.int32(coord_rt)

    'Find fundamental matrix and list of inliers after RANSAC algorithm'
    F, fundamentalStatus = cv2.findFundamentalMat(coord_left, coord_rt, cv2.FM_RANSAC, 20.0)
    print('\n\nFundamental matrix is:\n\n',F)
#    print(fundamentalStatus)#    print(fundamentalStatus)
print('\n\n############### Task 2 part 2 successfully completed.#####################\nTime taken for Task 2 part 2 = ',time.time()-t,' seconds')
########################## END PART 2 ####################################################
    

''''''''''''''''''''''''''''''''''''''' START PART 3 '''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''
print('\n\n##################### Starting execution of Task 2 part 3 ##########################################')

'Select the inliers and convert the inliers to a 1D list'
# Select only inlier points
'convert the inliers to a 1D list'
# Select only inlier points
'Select 10 random inliers after shuffling the array of inliers'
inliers_1 = coord_left[fundamentalStatus.ravel()==1].tolist()
inliers_left = random.sample(inliers_1,10)
inliers_left = np.array(inliers_left)

inliers_2 = coord_rt[fundamentalStatus.ravel()==1].tolist()
inliers_right = random.sample(inliers_2,10)
inliers_right = np.array(inliers_right)    

'Draw the epilines for 10 random inliers in both images'
task2_epi_right, task2_epi_left = findEpilines(tsucuba_left, tsucuba_right, inliers_left, inliers_right, F)
#task2_epi_right, task2_epi_lt = findEpilines(tsucuba_right, tsucuba_left, inliers_right, inliers_left, F)
writeImage(task2_epi_right, "task2_epi_right")
writeImage(task2_epi_left, "task2_epi_left")
print('\n\nImages task2_epi_right.jpg and task2_epi_left.jpg succesfully written to disk.')
##
print('\n\n############### Task 2 part 3 successfully completed.#####################\nTime taken for Task 2 part 3 = ',time.time()-t,' seconds')
################################ END PART 3 ####################################



########################### START PART 4 #########################################
print('\n\n##################### Starting execution of Task 2 part 4 ##########################################')
imgleft = cv2.imread('data/tsucuba_left.png', 0)
imgright = cv2.imread('data/tsucuba_right.png', 0)

'Create the object of stereo BM for computing disparity'
task2_stereo = cv2.StereoBM_create()
task2_disparity = task2_stereo.compute(imgleft, imgright)/3
writeImage(task2_disparity, "task2_disparity")
#plt.imshow(task2_disparity, 'gray')
#plt.savefig('./results/task2_disparity.jpg')
print('\n\n******************** Image task2_disparity successfully written to drive **************************')
print('\n\n############### Task 2 part 4 successfully completed.#####################\nTime taken for Task 2 part 4 = ',time.time()-t,' seconds')
