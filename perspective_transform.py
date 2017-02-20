
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import os.path as path


def region_of_interest(image, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(image, dtype='uint8')

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image.astype('uint8'), mask)
    return masked_image


def poly_line_f(points):
    """
    Given a course set of points return a function describing the best fit line.
    reference: http://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting
    """
    if len(points) == 0:
        return None

    points = np.array(points)
    # get x and y vectors
    x = points[:, 0]
    y = points[:, 1]

    # calculate polynomial
    z = np.polyfit(x, y, 1)
    f = np.poly1d(z)

    return f


def slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)


def y_intercept(x1, y1, m):
    return y1 - m * x1


def solve_for_x_given_y(line_f, y):
    assert len(line_f.coeffs) == 2, "line function expected to define simple line"
    return (y - line_f.coeffs[1]) / line_f.coeffs[0]


def line_intersection(line1, line2):
    """
    Identify the x coordinate where the lines intersect
    Reference: http://stackoverflow.com/questions/36686270/how-to-find-point-of-intersection-of-two-line-segments-in-python
    :param line1:
    :param line2:
    :Return x coordinage of intersection
    """
    a, b = list(line1)
    c, d = list(line2)
    x_intersect = (d - b) / (a - c)
    y_intersect = line1(x_intersect)
    return x_intersect, y_intersect


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    """

    :param image: `img` should be the output of a Canny transform.
    :param rho:
    :param theta:
    :param threshold:
    :param min_line_len:
    :param max_line_gap:
    """
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                           minLineLength=min_line_len, maxLineGap=max_line_gap)


def simple_quad(h, w, h_div=0.6, bw_div=0.08, tw_div=0.46, w_shift_div=0.02):
    """
    Returns for vertices for symmetrical quadrilateral within the provided rectangle.
    h: rectangle height
    w: rectangle width
    h_div: percent of rectangle height to size quad height
    bw_div: percent of rectangle width to size quad bottom insets
    tw_div: percent of rectangle width to size quad top insets
    w_shift_div: percent of rectangle width to shift entire quad left or right from center

    Return: 4 vertices as an arry of one numpy array containing the 4 vertices
    """
    w_shift = w * w_shift_div # shift polygon distance if lanes not perfectly centered

    by = h  # bottom y
    ty = h * h_div  # top y
    ltx = w * tw_div + w_shift # left top x
    rtx = w - w * tw_div + w_shift # right top x
    lbx = w * bw_div + w_shift # left bottom x
    rbx = w - w * bw_div + w_shift # right bottom x
    return [np.array([[lbx, by], [ltx, ty], [rtx, ty], [rbx, by]], dtype=np.int32)]


def enhanced_line(x1, y1, x2, y2, count=10, y_max=None):
    """
    Given a line defined by the points (x1, y1) and (x2, y2),
    enhance the line definition with additional points along the
    slope between those points.
    x1, y1: first point
    x2, y2: second point
    count: number of points to add
    y_max: if defined, also add a point at y_max along the same line
    """
    m = slope(x1, y1, x2, y2)
    b = y_intercept(x1, y1, m)
    points = []
    for sx in np.arange(x1, x2, (x2 - x1) / count):
        points.append((sx, m * sx + b))

    if y_max is not None and len(points) > 0:
        points.append((((y_max - b) / m), y_max))

    return points


def connected_line_functions(lines, h, w):
    w_mid = w / 2  # center of image from left to right
    left_bin = []  # collect left side points
    right_bin = []  # collect right side points
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = slope(x1, y1, x2, y2)
            if m <= 0 and x1 <= w_mid and x2 <= w_mid and abs(m) > 0.5:
                left_bin.extend(enhanced_line(x1, y1, x2, y2, 10, y_max=h))
            elif x1 > w_mid and x2 > w_mid and abs(m) > 0.5:
                right_bin.extend(enhanced_line(x1, y1, x2, y2, 10, y_max=h))

    # given set of points, define a best fit line for left and right lines
    left_line_f = poly_line_f(left_bin)
    right_line_f = poly_line_f(right_bin)

    return left_line_f, right_line_f


def find_quad_points(image):
    [h, w] = image.shape
    roi_image = region_of_interest(image, simple_quad(h, w))
    lines = hough_lines(roi_image,
                        rho=1,
                        theta=np.pi/90,
                        threshold=20,
                        min_line_len=10,
                        max_line_gap=40)

    left, right = connected_line_functions(lines, h, w)

    assert left is not None, "left line could not be identified"
    assert right is not None, "right line could not be identified"

    # define the quad points as the left, right lanes intersected by top and bottom horizontal lines
    x_intersect, y_intersect = line_intersection(left, right)
    left_top_y = int(y_intersect + 5)
    right_top_y = left_top_y
    left_top_x = int(solve_for_x_given_y(left, left_top_y))
    right_top_x = int(solve_for_x_given_y(right, right_top_y))
    left_bottom_y = int(h - 15)
    left_bottom_x = int(solve_for_x_given_y(left, left_bottom_y))
    right_bottom_y = int(left_bottom_y)
    right_bottom_x = int(solve_for_x_given_y(right, right_bottom_y))

    return np.array([(left_bottom_x, left_bottom_y),
                     (left_top_x, left_top_y),
                     (right_top_x, right_top_y),
                     (right_bottom_x, right_bottom_y)])


def demo_transform(calibration_image_names, road_image_names, output_folder, shape=(720, 1280), xct=9, yct=6):
    """
    run all parts of the pipeline and write sample images to ouput folder
    :param calibration_image_names: list of images to use for camera calibration
    :param road_image_names: list of images to apply pipeline to
    :param output_folder: folder to write output images to
    :param shape: expected shape of input images
    :param xct: expected corners in x direction for camera calibration
    :param yct: expected corners in y direction for camera calibration
    """
    import camera_calibration as cc
    import pipeline as pl
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

        # create threshholded binary image
        binary_image = pl.combined_binary(corrected_image)

        quad_points = find_quad_points(binary_image)
        quad_image = np.array(corrected_image, copy=True)
        cv2.line(quad_image, tuple(quad_points[0]), tuple(quad_points[1]), color=[255, 0, 0], thickness=4)
        cv2.line(quad_image, tuple(quad_points[1]), tuple(quad_points[2]), color=[255, 0, 0], thickness=4)
        cv2.line(quad_image, tuple(quad_points[2]), tuple(quad_points[3]), color=[255, 0, 0], thickness=4)
        cv2.line(quad_image, tuple(quad_points[3]), tuple(quad_points[0]), color=[255, 0, 0], thickness=4)
        # cv2.polylines(quad_image, quad_points, True, color=[255, 0, 0], thickness=4)
        # for point in quad_points:
        #     cv2.circle(quad_image, tuple(point), 5, color=[255, 0, 0])
        pl.save_single_example('/'.join([output_folder, "srcquad_{}".format(base_name)]),
                               'src quad', quad_image, cmap='jet')




        # save_full_example('/'.join([output_folder, "pipeline_{}".format(base_name)]),
        #                   cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
        #                   cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB),
        #                   binary_image)


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
    road_pattern = '/'.join([road_folder, 'straight_{}'.format(pattern)])
    road_names = glob.glob(road_pattern)
    demo_transform(calibrate_names, road_names, output_folder, xct=xct, yct=yct)


if __name__ == "__main__":
    main()

