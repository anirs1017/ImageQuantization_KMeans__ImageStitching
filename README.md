# ImageQuantization_KMeans__ImageStitching

<br><b> Task 1:  Find similarity between two images, match their features and warp one image with respect to the other based on the homography between the two images.</b>
<br>This task finds the similarity between two images based on their features and keypoints that are found using Scale Invariant Feature Transform (SIFT).
<br>Further, we find the best matches between these keypoints using K-Nearest Neighbour algorithm for nearest two neighbours (k=2) and draw the corresponding matches of all keypoints in both images.
<br>Next, we compute the homography matrix H using RANSAC method of comparison for all the keypoints in the first image to the second image. This computation gives us a projective space of a list of all inliers – points that are very close to the projective lines and relates the transformation between two planes.
<br><br>We visualize the matches by drawing 10 random matches using only inliers.
<br>Finally, we create a warping of the first image w.r.t the second image in a panoramic form.

<br><br><br><br><b> Task 2: Find the epipolar lines and epipoles for given two different images of the same frame.</b>
<br>First, we find the keypoints in two images, and compute which features in both images are similar. Once, we have found that, the corresponding matching features are matched using lines and the image is saved.
<br>Next, to calculate the amount of epipolarity between two images, it is necessary to calculate the fundamental matrix F to draw inference about the translation and rotation of two images, also with their intrinsic parameters. This is provided by F.
<br>After calculating F, we search for 10 random inlier epipolar points for both images. For each epipolar point in the left image, we compute the epiline and draw it on the right image and do it likewise for the right image and draw the epiline on the left image.
<br>While taking a photo of a 3D space by a single camera, the depth information of the 3D space is lost when it is mapped to a 2D space on the camera’s sensor. Hence, epipolarity helps us in finding the depth of a subject on a 2D frame.
<br>Finally, to find the depth and disparity of the images, we calculate the disparity map for both of them that gives us the depth information about the subjects in the frame and infer the disparities in the two images.

<br><br><br><br><b> Task 3: Implement the K-means algorithm for clustering any random unarranged dataset. The metric used in this project is the Euclidean distance as the distance function.</b>
<br>We are provided with a random dataset and a set of three random centroids for that dataset. Now, we compute the distances of all samples in the dataset with the centroids and calculate the minimum of the Euclidean distances of each centroid for every point. 
<br>Subsequently, the samples are clustered based on the centroid they are closest to. We repeat the same algorithm for each centroid and update it by taking the average of the new clustered datasets. That way, we go on clustering the entire dataset and will be left with all data points clustered properly.
