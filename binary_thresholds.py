
import numpy as np
import cv2


def abs_sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Apply sobel threshhold to image.
    Reference: Udacity Advanced Lane Finding - (21) Applying Sobel
    :param image: input image
    :param orient: orientation axis (x or y)
    :param sobel_kernel: kernel size to use, should be odd number
    :param thresh: pixel intensity threshhold
    :return: binary threshholded image
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    a = 1 if orient == 'x' else 0
    b = 1 if orient == 'y' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, a, b, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scalsqed gradient magnitude
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary


def magnitude_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    """
    Apply magnitude threshhold to image.
    Reference: Udacity Advanced Lane Finding - (22) Magnitude of the Gradient
    :param image: input image
    :param sobel_kernel: kernel size (should be odd)
    :param thresh: intensity threshhold
    :return: binary output image based on gradient magnitude
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) Take the gradient in x and y separately
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(abs_sobelx * abs_sobelx + abs_sobely * abs_sobely)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    scale_factor = np.max(abs_sobelxy)/255
    scaled_sobelxy = (abs_sobelxy/scale_factor).astype(np.uint8)

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobelxy)
    binary_output[(scaled_sobelxy >= thresh[0]) & (scaled_sobelxy <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def direction_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Apply direction of the gradient to image.
    Reference: Udacity Advanced Lane Finding - (23) Direction of the Gradient
    :param img: input image
    :param sobel_kernel: kernel size (should be odd)
    :param thresh: arctan2 range
    :return: binary image after direction threshold applied
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return binary_output


def color_binary(image, thresh=(170, 255)):
    """
    use saturation color channel to identify likely lane lines
    :param image: corrected image
    :param thresh: color threshold
    :return: binary image based on saturation
    """
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return s_binary


def combined_binary(image, gradx_kernel=3, grady_kernel=3, mag_kernel=3, dir_kernel=3,
                    gradx_thresh=(0, 255), grady_thresh=(0, 255), mag_thresh=(30, 100), dir_thresh=(0.7, 1.3),
                    color_thresh=(170, 255)):
    """
    use combination of all binary lane line identification methods to give best guess of lanes
    :param image: corrected image
    :param gradx_kernel: gradient kernel in x direction sobel
    :param grady_kernel: gradient kernel in y direction sobel
    :param mag_kernel: magnitude kernel
    :param dir_kernel: direction kernel
    :param gradx_thresh: pixel threshold for gradient on x sobel
    :param grady_thresh: pixel threshold for gradient on y sobel
    :param mag_thresh: threshold for magnitude
    :param dir_thresh: threshold for direction
    :param color_thresh: saturation threshold
    :return: combined result of all binary approaches
    """
    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(image, orient='x', sobel_kernel=gradx_kernel, thresh=gradx_thresh)
    grady = abs_sobel_threshold(image, orient='y', sobel_kernel=grady_kernel, thresh=grady_thresh)
    magnitude_bin = magnitude_threshold(image, sobel_kernel=mag_kernel, thresh=mag_thresh)
    direction_bin = direction_threshold(image, sobel_kernel=dir_kernel, thresh=dir_thresh)
    color_bin = color_binary(image, color_thresh)

    combined = np.zeros_like(direction_bin)
    combined[((gradx == 1) & (grady == 1) & (color_bin == 1)) | ((magnitude_bin == 1) & (direction_bin == 1))] = 1

    return combined

