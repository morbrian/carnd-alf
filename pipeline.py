
import numpy as np
import cv2
import glob
import os
import os.path as path
import matplotlib.pyplot as plt
import camera_calibration as cc


def save_single_example(output_image_name, title, image, cmap='jet'):
    """
    save the image to the output file
    :param output_image_name:
    :param title: title of image
    :param image: image data
    :param cmap: color map
    """
    fig = plt.figure()
    subplot = plt.subplot(1, 1, 1)
    subplot.axis('off')
    subplot.set_title(title)
    subplot.imshow(image, cmap=cmap)
    plt.savefig(output_image_name, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("saved to: {}".format(output_image_name))


def save_full_example(output_image_name, original_image, corrected_image, binary_image):
    """
    create a single figure of all images in a row and save to output name
    :param output_image_name: output image file name
    :param original_image: first image on left
    :param corrected_image: second image
    :param binary_image: third image
    """
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)

    subplot = plt.subplot(1, 3, 1)
    subplot.axis('off')
    subplot.set_title('original')
    plt.imshow(original_image)

    subplot = plt.subplot(1, 3, 2)
    subplot.axis('off')
    subplot.set_title('corrected')
    plt.imshow(corrected_image)

    subplot = plt.subplot(1, 3, 3)
    subplot.axis('off')
    subplot.set_title('binary')
    plt.imshow(binary_image, cmap='gray')

    plt.savefig(output_image_name, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("saved to: {}".format(output_image_name))


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


def demo_pipeline(calibration_image_names, road_image_names, output_folder, shape=(720, 1280), xct=9, yct=6):
    """
    run all parts of the pipeline and write sample images to ouput folder
    :param calibration_image_names: list of images to use for camera calibration
    :param road_image_names: list of images to apply pipeline to
    :param output_folder: folder to write output images to
    :param shape: expected shape of input images
    :param xct: expected corners in x direction for camera calibration
    :param yct: expected corners in y direction for camera calibration
    """
    if not path.exists(output_folder):
        os.makedirs(output_folder)

    object_points, image_points, pattern_found = \
        cc.calibration_points(calibration_image_names, xct=xct, yct=yct)
    ret, mtx, dist, rvecs, tvecs = cc.calibrate_camera(object_points, image_points, shape)

    for i, file_name in enumerate(road_image_names):
        base_name = path.split(file_name)[1]
        original_image = cv2.imread(file_name)

        # undistort the image
        corrected_image = cc.undistort_image(original_image, mtx, dist, mtx)
        save_single_example('/'.join([output_folder, "corrected_{}".format(base_name)]),
                            "corrected", cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))

        # create threshholded binary image
        binary_image = combined_binary(corrected_image)
        save_single_example('/'.join([output_folder, "binary_{}".format(base_name)]),
                            "binary", binary_image, cmap='gray')

        save_full_example('/'.join([output_folder, "pipeline_{}".format(base_name)]),
                          cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                          cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB),
                          binary_image)


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-c', '--calibrate_folder', dest='calibrate_folder', default='./camera_cal',
                      help="path to folder of calibration images to use.")
    parser.add_option('-r', '--road_folder', dest='road_folder', default='./test_images',
                      help="path to folder of calibration images to use.")
    parser.add_option('-o', '--output_folder', dest='output_folder', default='./output_folder',
                      help="output folder to hold examples of images during process.")
    parser.add_option('-p', '--pattern', dest='pattern', default='*.jpg',
                      help="filename pattern to match all files to use for callibration.")
    parser.add_option('-x', '--xct', dest='xct', default='9',
                      help="expected number of x corners.")
    parser.add_option('-y', '--yct', dest='yct', default='6',
                      help="expected number of y corners.")

    options, args = parser.parse_args()
    calibrate_folder = options.calibrate_folder
    road_folder = options.road_folder
    output_folder = options.output_folder
    xct = int(options.xct)
    yct = int(options.yct)
    pattern = options.pattern

    # road transform demo
    calibrate_pattern = '/'.join([calibrate_folder, pattern])
    calibrate_names = glob.glob(calibrate_pattern)
    road_pattern = '/'.join([road_folder, pattern])
    road_names = glob.glob(road_pattern)
    demo_pipeline(calibrate_names, road_names, output_folder, xct=xct, yct=yct)


if __name__ == "__main__":
    main()

