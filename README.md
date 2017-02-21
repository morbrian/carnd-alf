# Advanced Lane Finding Project

This is project 4 in the first third of the Udacity Self Driving Car Nanodegree.

The purpose of the project is to dive deeper into image processing and computer vision
strategies for identifying lane lines in images taken from a front facing dash mounted 
camera in vehicle driving down roads with painted lane lines.

## Write Up

This is document is the write up.
 
## Camera Calibration

The camera calibration code is available in [camera_calibration.py](./camera_calibration.py).

The calibration process is run once at program initialization time on each of the chess board images
found in the folder [camera_cal](./camera_cal). The following basic sequence of steps is followed
in order to calculate the calibration data.

        for each chessboard image: (camera_calibrtion.py:calibration_points line:29)
            identify corner points (camera_calibration.py:corner_points line:10)
            use points to calibrate camera (camera_calibration.py:calibrate_camera line:61)
            
The calibration process provides the `matrix` and `distance` array parameters to be used for
undistorting images taken by the same camera.

        corrected_image = undistort_image(image, mtx, dist, mtx)
        (camera_calibration.py:undistort_image line:72)

Below is an example of an original chessboard image (left), followed by an undistorted chessboard
with corners identified (right).

![calibration_image][calibration_image]
            
           
## Distortion Correction

Using the same matrix an distance parameters calculated during *Camera Calibration* we are able to
undistort any arbitrary image produced by the camera. This next image demonstrates an undistorted road image.

![undistort_image][undistort_image]

## Binary Image Analysis

We used a combination filter to highlight the lane line locations based on various characteristics we
further describe in this section. (pipeline.py:combined_binary line:167)

We used the `Sobel` algorithm to identify clusters of pixels likely to be lines based
 on thresholded x and y gradient values. (pipeline.py:abs_sobel_threshold line:60)
 
We converted the BGR image to an HLS image, and used the saturation channel to better identify
the yellow and white painted lines. (pipeline.py:color_binary line:149)

We produced a direction binary using `arctan2` on the sobel gradients and filtered on radian values 
within a threshold range. (pipeline.py:direction_threshold line:120)

We produced a magnitude binary to identify likely pixels by the magnitude of the sobel gradients and
filtered these on a threshold range.

Our `combined_binary` performs a conditional check on all of these, turning pixels on as white
where lines are likely to be, and to black otherwise.

The sequence below depicts the steps of the process so far for two samples, with the original on the left
followed by the undistorted image, and then the binary result on the right.

![binary_sequence1_image][binary_sequence1_image]
![binary_sequence2_image][binary_sequence2_image]

## Perspective Transform

Our perspective transform requires an initialization before it can reliably transform an arbitrary
image. The input image used for initialization should be hand picked and it must meet the following
 criteria:
 
 1. Should be a binary image output from the process above.
 2. Should be an image with straight lane lines on a flat road.
 
The initialzation process seeks to find the `src` and `dst` points which can be used for all future
transforms for the same camera mount location.

The process we used for identifying `src` and `dst` points for the image is very much like what we
used in Project-1 of the Udacity CarND. (perspective_transform.py:find_quad_points line:207)

1. We use the **Hough Lines** algorithm to discover the line segments
2. From the segment points we can identify functions for the left and right lane lines.
3. Using the lanes as sides of a trapezoid, we draw to parallel horizontal lines to complete the polygon.
4. We then produce a rectangle similar in height to the image as our destination polygon.

The steps above produce our `src` points in the form of the trapezoid, and our `dst` points in the
form of a rectangle.

We then have what we need to apply the cv2 `getPerspectiveTransform`, providing the Matrix input
 for a `warpPerspectiveTransorm` on any image later in the pipeline. 
 (perspective_transform.py:perspective_transform line:253)
 
Below is an example sequence of this initialization process on both of the provided straight line road images:

![perspective_transform1_image][perspective_transform1_image]
![perspective_transform2_image][perspective_transform2_image]

## Lane Line Fitting with Polynomial

Although we were able to successfully find the lane lines and fit them to a linear equation earlier
using a hough line algorithm when we initialized our perspective transform, we needed a more holistic
approach to handle curved lines.

We performed a significant amount of copy/paste from module 33 of the Udacity Advanced Lane Finding lesson.
(code in finding_lines.py)

Using a histogram to discover the likely start of the lane lines along the bottom of the image. The histogram
peaks indicate where a large number of pixels are turned on in the column, making it a likely lane line origin.

From the starting point, we use a small rectangular window to move up the image form the lane line origins.
The algorithm looks for active pixels and records them. If enough are found it recenters the window on
 the new center to follow the bulk of pixels along a line.
 
Once the search is completed, the algorithm uses the found pixels on the left and right to fit
a poly line through each group of points.

The image below visualizes the search windows (green) along the lane lines (red and blue) with the
fitted polylines (yellow).

![sliding_search][sliding_search]

## Lane Curvature

TODO

## Lane Line Result Visualization

TODO

## Pipeline Video

TODO

## Implementation Discussion

TODO


[//]: # (Image References)

[calibration_image]: ./output_folder/example_calibration3.jpg "calibration_image"
[undistort_image]: ./output_folder/corrected_test1.jpg "corrected_image"
[binary_sequence1_image]: ./output_folder/pipeline_binary_process_test3.jpg "binary_sequence1_image"
[binary_sequence2_image]: ./output_folder/pipeline_binary_process_test4.jpg "binary_sequence2_image"
[perspective_transform1_image]: ./output_folder/perspective_process_straight_lines1.jpg "perspective_transform1_image"
[perspective_transform2_image]: ./output_folder/perspective_process_straight_lines2.jpg "perspective_transform2_image"
[sliding_search]: ./output_folder/pipeline_sliding_search_test3.jpg "sliding_search"


