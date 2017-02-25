# Advanced Lane Finding Project

This is project 4 in the first third of the Udacity Self Driving Car Nanodegree.

The purpose of the project is to dive deeper into image processing and computer vision
strategies for identifying lane lines in images taken from a front facing dash mounted 
camera in vehicle driving down roads with painted lane lines.j

Any results in the output folder may be reproduced by running each of the main methods with default values.

        python camera_calibration.py
        python perspective_transform.py
        python pipeline.py

## Write Up

This is document is the write up.
 
## Camera Calibration

The camera calibration code is available in [camera_calibration.py](./camera_calibration.py).

The calibration process is run once at program initialization time on each of the chess board images
found in the folder [camera_cal](./camera_cal).

At [camera_calibrtion.py:calibration_points](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/camera_calibration.py#L50-L56)` 
the calibration images are looped over.
The combined image point outputs from this process are used by the function [camera_calibration.py:calibrate_camera](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/camera_calibration.py#L61)
            
The calibration process provides the `matrix` and `distance` array parameters to be used for
undistorting images taken by the same camera.

corrected_image = undistort_image(image, mtx, dist, mtx) [camera_calibration.py:undistort_image](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/camera_calibration.py#L72)

Below is an example of an original chessboard image (left), followed by an undistorted chessboard
with corners identified (right).

![calibration_image][calibration_image]
            
           
## Distortion Correction

Using the same matrix an distance parameters calculated during *Camera Calibration* we are able to
undistort any arbitrary image produced by the camera. This next image demonstrates an undistorted road image.

![undistort_image][undistort_image]

## Binary Image Analysis

We used a combination filter to highlight the lane line locations based on various characteristics we
further describe in this section. [binary_thresholds.py:combined_binary](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/binary_thresholds.py#L113)

We used the `Sobel` algorithm to identify clusters of pixels likely to be lines based
 on thresholded x and y gradient values. [binary_thresholds.py:abs_sobel_threshold](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/binary_thresholds.py#L6)
 
We converted the BGR image to an HLS image, and used the saturation channel to better identify
the yellow and white painted lines. [binary_thresholds.py:color_binary](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/binary_thresholds.py#L95)

We produced a direction binary using `arctan2` on the sobel gradients and filtered on radian values 
within a threshold range. [binary_thresholds.py:direction_threshold](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/binary_thresholds.py#L66)

We produced a magnitude binary to identify likely pixels by the magnitude of the sobel gradients and
filtered these on a threshold range.

Our [binary_thresholds.py:combined_binary](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/binary_thresholds.py#L113) function performs a conditional check on all of these, turning pixels on as white
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

[perspective_transform.py:find_quad_points](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/perspective_transform.py#L209-L252)
performs the initialization process using the input image to find the `src` and `dst` points which can be used for all future
transforms for the same camera mount location.

The process defined in `find_quad_points` is very much like what we
used in Project-1 of the Udacity CarND.

1. We use the [Hough Lines algorithm](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/perspective_transform.py#L222) to discover the line segments
2. From the segment points we can identify [functions for the left and right lane lines](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/perspective_transform.py#L229).
3. Using the lanes as sides of a trapezoid, we draw to parallel horizontal lines to [complete the polygon](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/perspective_transform.py#L235-L248).
4. We then [produce a rectangle](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/perspective_transform.py#L250) similar in height to the image as our destination polygon.

The steps above produce our `src` points in the form of the trapezoid, and our `dst` points in the
form of a rectangle.

We then have what we need to apply the cv2 `getPerspectiveTransform`, providing the Matrix input
 for a `warpPerspectiveTransorm` on any image [later in the pipeline](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/perspective_transform.py#L256). 
 
Below is an example sequence of this initialization process on both of the provided straight line road images:

![perspective_transform1_image][perspective_transform1_image]
![perspective_transform2_image][perspective_transform2_image]

## Lane Line Fitting with Polynomial

Although we were able to successfully find the lane lines and fit them to a linear equation earlier
using a hough line algorithm when we initialized our perspective transform, we needed a more holistic
approach to handle curved lines.

We performed a significant amount of copy/paste from module 33 of the Udacity Advanced Lane Finding lesson.
(code in [finding_lines.py](./finding_lines.py))

[Using a histogram](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/finding_lines.py#L192) to discover the likely start of the lane lines along the bottom of the image. The histogram
peaks indicate where a large number of pixels are turned on in the column, making it a likely lane line origin.

From the starting point, we use a small rectangular window to move up the image form the lane line origins.
The algorithm looks for active pixels and records them. [If enough are found it recenters](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/finding_lines.py#L246-L249) the window on
 the new center to follow the bulk of pixels along a line.
 
Once the search is completed, the algorithm uses the found pixels on the left and right to fit
a poly line through each group of points.

The image below visualizes the search windows (green) along the lane lines (red and blue) with the
fitted polylines (yellow).

![sliding_search][sliding_search]

## Lane Curvature

Our lane curvature calculations are in [finding_lines.py](./finding_lines.py). We first used the polynomial fit line output 
of the lane fitting algorithm as input to the [finding_lines.py:radius_curvature_pixel_space](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/finding_lines.py#L302)
function to produce the curved lines in pixel space. Then based on advice in the lesson we get the
real world curvature in meters in [finding_lines.py:radius_curvature_meters](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/finding_lines.py#L318).

## Lane Line Result Visualization

The road ahead is highlighted in green in the image sample below, and the curvature and distance
from the lane center are both labeled in the top left corner of the image in meters.

The code for this visualization is at [finding_lanes.py:visualize_road_ahead](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/finding_lines.py#L42).

![road_ahead][road_ahead]

## Pipeline Video

The two important entry points for the video pipeline code are:

1. [pipeline.py:process_video](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/pipeline.py#L194)
    
    This performs the [initial calibration an inititalization steps](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/pipeline.py#L213-L217),
     then [loads the video](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/pipeline.py#L221-L222).

2. [pipeline.py:apply_pipeline](https://github.com/morbrian/carnd-alf/blob/57a63092b970a89ca26dee61001739e96116a758/pipeline.py#L117-L142)

    This performs each step of the video frame processing, with a small amount of extra codej
    to help with debugging support.

The pipeline performed reasonably well on the project video. Some extra tuning of the HLS saturation threshhold
was required to cross the first light colored overpass. Most of the video wobbles a little on the bottom right
as the lane line dashes come and go. There is a minor increase in wobbling after the second over pass due
to the shadows, but it does not appear to creat significant risk to how the vehicle would drive.

Watch the video here: [project_video_output](project_video_output)

## Implementation Discussion

The pipeline is sensitive to changes in road color and lighting changes. Most of these problems can
easily be overcome in isolation simply by tuning the parameters for one or more of the binary images
we combine. However, getting one category of image tuned correctly can cause issues in another category
of image.

One idea I was considering but did not take the time to implement was to directly categorize the images
but the priority of tuning parameters used. Some example categories might have been "Light Road", "Dark Road",
"Scattered Shadows", etc. Then Associate these road categories with the tuning parameters that I found to
work best for images in each category. Finally, apply a machine learning technique to identify the best
parameters sets for each image before processing it.


[//]: # (Image References)

[calibration_image]: ./output_folder/example_calibration3.jpg "calibration_image"
[undistort_image]: ./output_folder/corrected_test1.jpg "corrected_image"
[binary_sequence1_image]: ./output_folder/pipeline_binary_process_test3.jpg "binary_sequence1_image"
[binary_sequence2_image]: ./output_folder/pipeline_binary_process_test4.jpg "binary_sequence2_image"
[perspective_transform1_image]: ./output_folder/perspective_process_straight_lines1.jpg "perspective_transform1_image"
[perspective_transform2_image]: ./output_folder/perspective_process_straight_lines2.jpg "perspective_transform2_image"
[sliding_search]: ./output_folder/pipeline_sliding_search_test3.jpg "sliding_search"
[road_ahead]: ./output_folder/pipeline_road_ahead_test4.jpg "road_ahead"
[project_video_output]: ./output_folder/road_ahead_project_video.mp4 "road_ahead_video"

