
import numpy as np
import cv2
import glob
import os
import os.path as path
import matplotlib.pyplot as plt


def corner_points(image_name, xct=9, yct=6):
    # read in each image
    """
    Read mimage at image_name and return corner points.
    Reference: (udacity) https://youtu.be/lA-I22LtvD4
    :param image_name: path to image file to laod
    :param xct: expected count of x points
    :param yct: expected count of y points
    :return: found points
    """
    image = cv2.imread(image_name)

    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find the chessboard corners
    return cv2.findChessboardCorners(gray, (xct, yct), None)


def calibration_points(image_names, xct=9, yct=6):
    """
    Find calibration points for each input image.
    Reference: (udacity) https://youtu.be/lA-I22LtvD4
    :param image_names: list of image names to search
    :param xct: expected count of x points
    :param yct: expected count of y points
    :return:
    pattern_found will be of same size as input image_names,
    other outputs will exclude data when object not found
        object_points: expected object points array
        image_points: location of points in the image
        pattern_found: return value indicating handling if object was found, or False if not
    """
    object_points = []
    image_points = []
    pattern_found = []

    objp = np.zeros((yct * xct, 3), np.float32)
    objp[:, :2] = np.mgrid[0:xct, 0:yct].T.reshape(-1, 2)

    for file_name in image_names:
        ret, corners = corner_points(file_name, xct=xct, yct=yct)

        pattern_found.append(ret)
        if ret:
            image_points.append(corners)
            object_points.append(objp)

    return object_points, image_points, pattern_found


def calibrate_camera(object_points, image_points, shape):
    """
    Simple wrapper around cv2.calibrateCamera
    :param object_points: described object points
    :param image_points: associated points in image
    :param shape: shape of 2D image
    :return: ret, mtx, dist, rvecs, tvecs (same as cv2 function)
    """
    return cv2.calibrateCamera(object_points, image_points, shape, None, None)


def undistort_image(image, mtx, dist, newMtx):
    """
    Simple wrapper around cv2.undistort
    :param image: image to undistort and return
    :param mtx: camera matrix
    :param dist: distance coefficients
    :param newMtx: new matrix
    :return: corrected image
    """
    return cv2.undistort(image, mtx, dist, None, newMtx)


def save_example(output_image_name, original_image, undistorted_image):
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)

    subplot = plt.subplot(1, 2, 1)
    subplot.axis('off')
    subplot.set_title('original')
    plt.imshow(original_image)

    subplot = plt.subplot(1, 2, 2)
    subplot.axis('off')
    subplot.set_title('undistorted')
    plt.imshow(undistorted_image)

    plt.savefig(output_image_name, bbox_inches='tight', dpi=150)
    print("saved to: {}".format(output_image_name))


def demo_camera_calibration(image_names, output_folder, shape=(720, 1280), xct=9, yct=6):
    """
    Display each of the images with the corner points drawn.
    :param image_names: list of image names to evaluate
    :param output_folder: folder to save demo images after processing
    :param shape: expected shape of each image in the list (they should all be same size)
    :param xct: expected count of x points
    :param yct: expected count of y points
    """
    if not path.exists(output_folder):
        os.makedirs(output_folder)

    object_points, image_points, pattern_found = \
        calibration_points(image_names, xct=xct, yct=yct)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(object_points, image_points, shape)

    skip = 0
    for i, file_name in enumerate(image_names):
        if pattern_found[i]:
            base_name = path.split(file_name)[1]
            original_image = cv2.imread(file_name)
            corners = image_points[i - skip]
            ret = pattern_found[i]

            # draw corners on image for demo purposes
            corner_image = cv2.drawChessboardCorners(np.array(original_image, copy=True), (xct, yct), corners, ret)

            # create an undistorted version of image (now with corners drawn)
            undistorted_image = undistort_image(corner_image, mtx, dist, mtx)

            # save an example of how the work looks at each step
            output_file_name = '/'.join([output_folder, "example_{}".format(base_name)])
            save_example(output_file_name, original_image, undistorted_image)
        else:
            skip += 1


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-c', '--calibrate_folder', dest='calibrate_folder', default='./camera_cal',
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
    output_folder = options.output_folder
    xct = int(options.xct)
    yct = int(options.yct)
    pattern = options.pattern

    # calibration demo
    calibrate_pattern = '/'.join([calibrate_folder, pattern])
    calibrate_names = glob.glob(calibrate_pattern)
    demo_camera_calibration(calibrate_names, output_folder, xct=xct, yct=yct)


if __name__ == "__main__":
    main()

