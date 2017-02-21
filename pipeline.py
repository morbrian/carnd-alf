
import numpy as np
import cv2
import glob
import os
import os.path as path
import matplotlib.pyplot as plt
import camera_calibration as cc
import perspective_transform as pt
import binary_thresholds as bt
import finding_lines as fl


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


def demo_pipeline(calibration_image_names, road_image_names, straight_image_name, output_folder,
                  shape=(720, 1280), xct=9, yct=6):
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

    # camera calibration
    object_points, image_points, pattern_found = \
        cc.calibration_points(calibration_image_names, xct=xct, yct=yct)
    ret, mtx, dist, rvecs, tvecs = cc.calibrate_camera(object_points, image_points, shape)

    # perspective transform initialization
    ci = cc.undistort_image(cv2.imread(straight_image_name), mtx, dist, mtx)
    bi = bt.combined_binary(ci)
    src, dst = pt.find_quad_points(bi)

    for i, file_name in enumerate(road_image_names):
        base_name = path.split(file_name)[1]
        original_image = cv2.imread(file_name)

        # undistort the image
        corrected_image = cc.undistort_image(original_image, mtx, dist, mtx)
        save_single_example('/'.join([output_folder, "corrected_{}".format(base_name)]),
                            "corrected", cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))

        # create threshholded binary image
        binary_image = bt.combined_binary(corrected_image)
        save_single_example('/'.join([output_folder, "pipeline_binary_{}".format(base_name)]),
                            "binary", binary_image, cmap='gray')

        save_full_example('/'.join([output_folder, "pipeline_binary_process_{}".format(base_name)]),
                          cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                          cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB),
                          binary_image)

        transformed = pt.perspective_transform(binary_image, src, dst)
        save_single_example('/'.join([output_folder, "pipeline_transformed_{}".format(base_name)]),
                            "pipeline transformed", transformed, cmap='gray')

        # weighted = fl.sliding_window_search(transformed)
        # save_single_example('/'.join([output_folder, "pipeline_weighted_{}".format(base_name)]),
        #                     "pipeline weighted", weighted, cmap='jet')
        searched_image, nonzerox, nonzeroy, left_fit, right_fit, left_lane_inds, right_lane_inds = \
            fl.sliding_histo_search(transformed)
        fl.visualize_sliding_search('/'.join([output_folder, "pipeline_sliding_search_{}".format(base_name)]),
                                    "sliding search", transformed, searched_image, nonzerox, nonzeroy,
                                    left_fit, right_fit, left_lane_inds, right_lane_inds)


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-c', '--calibrate_folder', dest='calibrate_folder', default='./camera_cal',
                      help="path to folder of calibration images to use.")
    parser.add_option('-r', '--road_folder', dest='road_folder', default='./test_images',
                      help="path to folder of calibration images to use.")
    parser.add_option('-s', '--straight_image', dest='straight_image', default='./test_images/straight_lines1.jpg',
                      help="path to file with straight road to initialize perspective quads.")
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
    straight_image = options.straight_image

    # road transform demo
    calibrate_pattern = '/'.join([calibrate_folder, pattern])
    calibrate_names = glob.glob(calibrate_pattern)
    road_pattern = '/'.join([road_folder, pattern])
    road_names = glob.glob(road_pattern)
    demo_pipeline(calibrate_names, road_names, straight_image, output_folder, xct=xct, yct=yct)


if __name__ == "__main__":
    main()

