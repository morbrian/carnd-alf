
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import os.path as path
import binary_thresholds as bt

def save_full_example(output_image_name, corrected_image, quad_image, transform_image):
    """
    create a single figure of all images in a row and save to output name
    :param output_image_name: output image file name
    :param corrected_image: first image on left
    :param quad_image: second image
    :param transform_image: third image
    """
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)

    subplot = plt.subplot(1, 3, 1)
    subplot.axis('off')
    subplot.set_title('corrected')
    plt.imshow(corrected_image)

    subplot = plt.subplot(1, 3, 2)
    subplot.axis('off')
    subplot.set_title('quad points')
    plt.imshow(quad_image, cmap='jet')

    subplot = plt.subplot(1, 3, 3)
    subplot.axis('off')
    subplot.set_title('perspective transformed')
    plt.imshow(transform_image)

    plt.savefig(output_image_name, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("saved to: {}".format(output_image_name))


def region_of_interest(image, vertices):
    """
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
    """
    Returns the slope of a line intersecting points (x1, y1) and (x2, y2)
    """
    return (y2-y1)/(x2-x1)


def y_intercept(x1, y1, m):
    """
    Returns the y intecept for the line with slop `m` intersecting point (x1, y1)
    """
    return y1 - m * x1


def solve_for_x_given_y(line_f, y):
    """
    Return the x value for the line function `line_f` when y equals the given y.
    :param line_f: line function
    :param y: y value
    """
    assert len(line_f.coeffs) == 2, "line function expected to define simple line"
    return (y - line_f.coeffs[1]) / line_f.coeffs[0]


def line_intersection(line1, line2):
    """
    Identify the x coordinate where the lines intersect
    Reference: http://stackoverflow.com/questions/36686270/how-to-find-point-of-intersection-of-two-line-segments-in-python
    :param line1:
    :param line2:
    :Return coordinate of intersection (x, y)
    """
    a, b = list(line1)
    c, d = list(line2)
    x_intersect = (d - b) / (a - c)
    y_intersect = line1(x_intersect)
    return x_intersect, y_intersect


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Given a binary image, return the line definitions using Hough Lines algorithm.
    :param image: `img` should be a black and white binary image.
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
    Return: 4 vertices as an array of one numpy array containing the 4 vertices
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
    """
    Given a list of unrelated lines, attempt to group the points
    into left and right bins to create a single left line and single right line.
    :param lines: list of lines as point pairs
    :param h: height of space
    :param w: width of space
    :return: functions for left and right lines (left, right)
    """
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


def find_quad_points(image, s_ty_offset=40, s_by_offset=15, d_ty_offset=5):
    """
    Given a binary image with an assumed to be straigt road,
    discover the left and right lanes and uses these to identify
    a good trapezoidal shap to use as source points for prespective transforms.
    :param image: binary image
    :param s_ty_offset: top y offset
    :param s_by_offset: bottom y offset
    :param d_ty_offset: destination top y offset
    :return: suggested source and destination points for perspective transform.
    """
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
    left_top_y = int(y_intersect + s_ty_offset)
    right_top_y = left_top_y
    left_top_x = int(solve_for_x_given_y(left, left_top_y))
    right_top_x = int(solve_for_x_given_y(right, right_top_y))
    left_bottom_y = int(h - s_by_offset)
    left_bottom_x = int(solve_for_x_given_y(left, left_bottom_y))
    right_bottom_y = int(left_bottom_y)
    right_bottom_x = int(solve_for_x_given_y(right, right_bottom_y))

    src = np.array([(left_bottom_x, left_bottom_y),
                    (left_top_x, left_top_y),
                    (right_top_x, right_top_y),
                    (right_bottom_x, right_bottom_y)], dtype='float32')

    dst = np.array([src[0], (src[0][0], d_ty_offset), (src[3][0], d_ty_offset), src[3]], dtype='float32')

    return src, dst


def perspective_transform(image, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


def draw_quad(image, quad, color=[0, 255, 0], thickness=4):
    cv2.line(image, tuple(quad[0]), tuple(quad[1]), color=color, thickness=thickness)
    cv2.line(image, tuple(quad[1]), tuple(quad[2]), color=color, thickness=thickness)
    cv2.line(image, tuple(quad[2]), tuple(quad[3]), color=color, thickness=thickness)
    cv2.line(image, tuple(quad[3]), tuple(quad[0]), color=color, thickness=thickness)


def convert_binary_to_color(image):
    return cv2.cvtColor(image.astype('uint8') * 255, cv2.COLOR_GRAY2BGR)


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
        h, w, _ = corrected_image.shape

        # create threshholded binary image
        binary_image = bt.combined_binary(corrected_image)

        src, dst = find_quad_points(binary_image)
        quad_image = convert_binary_to_color(binary_image)
        draw_quad(quad_image, src, color=[255, 0, 0])
        draw_quad(quad_image, dst, color=[0, 255, 0])
        pl.save_single_example('/'.join([output_folder, "srcquad_{}".format(base_name)]),
                               'src quad', quad_image, cmap='jet')

        transformed = perspective_transform(corrected_image, src, dst)
        draw_quad(transformed, dst, color=[0, 0, 255])
        pl.save_single_example('/'.join([output_folder, "init_perspective_{}".format(base_name)]),
                               'initializing perspective transform', cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB), cmap='jet')

        save_full_example('/'.join([output_folder, "perspective_process_{}".format(base_name)]),
                          cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB),
                          quad_image,
                          cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))


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

