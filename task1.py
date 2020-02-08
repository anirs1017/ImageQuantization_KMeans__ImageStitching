# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:29:06 2018
@author: Aniruddha Sinha
UB Person Number = 50289428
UBIT = asinha6@buffalo.edu

"""
UBIT = '<asinha6>'; import numpy as np; np.random.seed(sum([ord(c) for c in UBIT]))

import cv2
import numpy as np
import random
import math
import time

'function to write the image to disk'
def writeImage(img, imageName):
    cv2.imwrite("results/" + imageName + ".jpg", img)

'Start function execution'
t = time.time()

'Read the two images, first in colour and then in grayscales'
left_img = cv2.imread('data/mountain1.jpg')
rt_img = cv2.imread('data/mountain2.jpg')

mountain1 = cv2.imread("data/mountain1.jpg", 0)
mountain2 = cv2.imread("data/mountain2.jpg", 0)

############################# PART 1 ##################################################
print('\n\n##################### Starting execution of Task 1 part 1 ##########################################')

''' The following code for computing SIFT features and drawing keypoints has been referenced and partly copied from '''
'''' https://docs.opencv.org/3.4.3/da/df5/tutorial_py_sift_intro.html '''
      
'Compute the keypoints and features of the two images'
sift = cv2.xfeatures2d.SIFT_create()
keypoints1 = sift.detect(mountain1, None)
keypoints2 = sift.detect(mountain2, None)

'Draw the keypoints on the images and write to disk'
task1_sift1 = cv2.drawKeypoints(left_img, keypoints1, None) 
task1_sift2 = cv2.drawKeypoints(rt_img, keypoints2, None)
writeImage(task1_sift1, "task1_sift1")
print('\n\n******************** Image task1_sift11 successfully written to drive **************************')
writeImage(task1_sift2, "task1_sift2")
print('\n\n******************** Image task2_sift2 successfully written to drive **************************')
print('\n\n############### Task 1 part 1 successfully completed.#####################\nTime taken for Task 1 part 1 = ',time.time()-t,' seconds')

############################# END PART 1 ################################################

########################## PART 2 ###################################################
print('\n\n##################### Starting execution of Task 1 part 2 ##########################################')
'Find the keypoints and the descriptors of the keypoints'      

''' The following code for computing the K Nearest Neighbours and drawing matches
 has been referenced and party copied from - 
 https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html '''

keypoint_mountain1, descriptor_left_img = sift.detectAndCompute(mountain1, None)
keypoint_mountain2, descriptor_rt_img = sift.detectAndCompute(mountain2, None)

'Using FLANN matching technique for finding K Nearest Neighbours'
FLANN_INDEX_KDTREE = 0

check_index_params = dict(algorithm = FLANN_INDEX_KDTREE, tress = 5)
check_search_params = dict(checks=100)

'''Create object for FLANN matching'''
doFlann = cv2.FlannBasedMatcher(check_index_params, check_search_params) 
NearestNeighborsMatches = doFlann.knnMatch(descriptor_left_img, descriptor_rt_img, k=2)


'Calculate the better neighbour of the two best matches of a keypoint of left image in the right image'
keepKeypoints = []
for m,n in NearestNeighborsMatches:
    if m.distance < 0.75*n.distance:
        keepKeypoints.append(m)

'Draw the matched keypoints of the two images'
task1_matches_knn = cv2.drawMatches(left_img, keypoint_mountain1, rt_img, keypoint_mountain2, keepKeypoints, None, flags=2)
writeImage(task1_matches_knn, "task1_matches_knn")
print('\n\n******************** Image task1_matches_knn successfully written to drive **************************')
print('\n\n############### Task 1 part 2 successfully completed.#####################\nTime taken for Task 1 part 2 = ',time.time()-t,' seconds')

########################## END PART 2 ####################################################

################################# PART 3 #################################################
# Draw the matches between 10 random points in the two images
#task1_matches_knn = cv2.drawMatchesKnn(mountain1, keypoint_mountain1, mountain2, keypoint_mountain2, NearestNeighborsMatches, None, **draw_parameters)
print('\n\n##################### Starting execution of Task 1 part 3 ##########################################')

''' The following code for finding Homography and computing the inliers has been referenced and 
party been copied from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html'''
      
MIN_NUM_MATCHES = 10

if len(keepKeypoints) > MIN_NUM_MATCHES:
    coord1 = np.array([ keypoint_mountain1[m.queryIdx].pt for m in keepKeypoints ])
    np.reshape(coord1, (-1,1,2))
    coord1 = np.float32(coord1)
    coord2 = np.array([ keypoint_mountain2[m.trainIdx].pt for m in keepKeypoints ])
    np.reshape(coord2, (-1,1,2))
    coord2 = np.float32(coord2)
    
    'Find homography matrix and list of inliers after RANSAC algorithm'
    H, homographyStatus = cv2.findHomography(coord1, coord2, cv2.RANSAC, 5.0)
#
    print('\n\nHomography matrix is',H)
#    print('Homography status', homographyStatus)

    print('\n\n############### Task 1 part 3 successfully completed.#####################\nTime taken for Task 1 part 3 = ',time.time()-t,' seconds')    
############################ END PART 3 ##########################################

########################## PART 4 ########################################
    print('\n\n##################### Starting execution of Task 1 part 4 ##########################################')
    'convert the inliers to a 1D list'
    maskOfMatches = homographyStatus.ravel().tolist()
else:
    print('\n\nMatches found are less than 10')
    maskOfMatches = None

'Select 10 random inliers after shuffling the array of inliers'    
tenMatches = np.where(np.array(maskOfMatches)==1)[0]
np.random.shuffle(tenMatches)
tenRandomMatches = tenMatches[1:10].tolist()
#matchesMask= np.array(maskOfMatches)
matchesMask = []

for pt in range(len(maskOfMatches)):
    if pt in tenRandomMatches:
        matchesMask.append(1)
    else:
        matchesMask.append(0)
#print('\n\n\n10 random inlier locations are:')
print(tenRandomMatches)        
#print(len(matchesMask), len(maskOfMatches), len(keepKeypoints))

'Draw the matched 10 random inliers in both images'
task1_matches = cv2.drawMatches(left_img, keypoint_mountain1, rt_img, keypoint_mountain2, keepKeypoints, None, (0, 0, 255), None, matchesMask, flags=2)
writeImage(task1_matches, "task1_matches")
print('\n\n******************** Image task1_matches successfully written to drive **************************')
print('\n\n############### Task 1 part 4 successfully completed.#####################\nTime taken for Task 1 part 4 = ',time.time()-t,' seconds')

########################### END PART 4 #########################################

########################### PART 5 ############################################
print('\n\n##################### Starting execution of Task 1 part 5 ##########################################')

''' The following code for computing the translation and rotation of image corners 
 and warping one image to the other using cv2.warpPerspective has been referenced from https://docs.opencv.org/3.4.3/da/d6e/tutorial_py_geometric_transformations.html
 and https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545'''      

'Read the shapes of the two images'      
mountain1_h, mountain1_w = mountain1.shape
mountain2_h, mountain2_w = mountain2.shape

'Save the corners of each image in a form of list of list'
points_m1 = np.float32([[0,0], [0,mountain1_h], [mountain1_w, mountain1_h], [mountain1_w,0]]).reshape(-1,1,2)
points_m2 = np.float32([[0,0], [0, mountain2_h], [mountain2_w, mountain2_h], [mountain2_w,0]]).reshape(-1,1,2)

'Compute the translation and rotation of each corner of left image w.r.t the right image using Homography matrix'
points_m1_modify = cv2.perspectiveTransform(points_m1,H)
#print('points_m1_modify',points_m1_modify)

'''Add the translated corners of the left image to the corners of the right image in the form of a 
list of list for each corner location on the canvas. We add alon axis=0, i.e. along the rows'''
all_mPoints = np.concatenate((points_m2, points_m1_modify), axis=0)

'''Find the minimum and maximum values from all the corners to create the size of the frame at a 
 distance of +-1 '''
[pano_xmin, pano_ymin] = np.int32(all_mPoints.min(axis=0).ravel() - 1.0)
[pano_xmax, pano_ymax] = np.int32(all_mPoints.max(axis=0).ravel() + 1.0)

'Calculate the left upper most and lower most corners of the frame'
transformationM = [-pano_ymin, -pano_xmin]

'Compute the Translated matrix for the new frame, w.r.t which our H will be translated'
translatedH = np.array([[1,0,transformationM[1]], [0,1,transformationM[0]], [0,0,1]])  

'warp the left image w.r.t the right image'
task1_pano = cv2.warpPerspective(left_img, translatedH.dot(H), (pano_xmax - pano_xmin, pano_ymax - pano_ymin))
task1_pano[transformationM[0]:mountain2_h + transformationM[0], transformationM[1]:mountain2_w + transformationM[1]] = rt_img
writeImage(task1_pano,"task1_pano")
print('\n\n******************** Image task1_pano successfully written to drive **************************')
print('\n\n############### Task 1 part 5 successfully completed.#####################\nTime taken for Task 1 part 5 = ',time.time()-t,' seconds')
