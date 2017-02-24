
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
from moviepy.editor import VideoFileClip
import imageio
imageio.plugins.ffmpeg.download()


class RoadAheadPipeline:
    """ container class for road ahead image processing """

    mtx = None  # matrix for camera calibration
    dist = None  # distance array for camera calibration
    src = None  # source points for transforms
    dst = None  # destination points for transforms
    M_transform = None  # transformation matrix for overhead perspective
    Minv_transform = None  # inverse transformation matrix for camera perspective
    left_fit = None  # left lane line polynomial
    right_fit = None  # right lane line polynomial
    left_curve_radius_meters = None  # left radius meters
    right_curve_radius_meters = None  # right radius meters
    off_center_meters = None  # number of meters to left (negative) or right (positive)

    def __init__(self):
        pass

    def calibrate_camera(self, calibration_image_names, xct=9, yct=6, shape=(720, 1280)):
        """
        Calibrate the camera using a list of sample chessboard images.
        :param calibration_image_names: list of paths to chessboard images
        :param shape: expected shape of input images
        :param xct: expected corners in x direction for camera calibration
        :param yct: expected corners in y direction for camera calibration
        """
        object_points, image_points, pattern_found = \
            cc.calibration_points(calibration_image_names, xct=xct, yct=yct)
        _, self.mtx, self.dist, _, _ = cc.calibrate_camera(object_points, image_points, shape)

    def initialize_perspective_transform(self, image):
        ci = cc.undistort_image(image, self.mtx, self.dist, self.mtx)
        bi = self.combined_binary_image(ci)
        self.src, self.dst = pt.find_quad_points(bi)
        self.M_transform = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv_transform = cv2.getPerspectiveTransform(self.dst, self.src)

    def first_video_frame_image(self, image):
        searched_image, left_fit, right_fit, left_lane_inds, right_lane_inds = \
            fl.sliding_histo_search(image)
        self.left_fit = left_fit
        self.right_fit = right_fit
        return searched_image, left_lane_inds, right_lane_inds

    def undistort_image(self, image):
        return cc.undistort_image(image, self.mtx, self.dist, self.mtx)

    def binary_image(self, image):
        return bt.combined_binary(image)

    def combined_binary_image(self, image):
        return bt.combined_binary(image)

    def perspective_transform_image(self, image):
        return pt.perspective_transform(image, self.M_transform)

    def fit_frame_image(self, image):
        ploty, (self.left_fit, self.right_fit), (left_fitx, right_fitx) = \
            fl.margin_search(image, self.left_fit, self.right_fit)
        self.left_curve_radius_meters, self.right_curve_radius_meters = \
            fl.radius_curvature_meters(ploty, left_fitx, right_fitx)
        self.off_center_meters = \
            fl.off_center_meters(image.shape[1], left_fitx[-1], right_fitx[-1])

    def average_curve_radius_meters(self):
        return (self.left_curve_radius_meters + self.right_curve_radius_meters) / 2.

    def road_ahead_image(self, image, transformed):
        return fl.visualize_road_ahead(image,
                                       transformed, self.left_fit, self.right_fit,
                                       self.average_curve_radius_meters(),
                                       self.off_center_meters, self.Minv_transform)

    def apply_pipeline(self, image):
        if self.left_fit is None or self.right_fit is None:
            self.first_video_frame_image(image)

        corrected_image = self.undistort_image(image)
        binary_image = self.binary_image(corrected_image)
        transformed = self.perspective_transform_image(binary_image)
        self.fit_frame_image(transformed)
        road_ahead_image = \
            self.road_ahead_image(corrected_image, transformed)
        return road_ahead_image


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


def demo_pipeline(calibration_image_names, road_image_names, straight_image_name, output_folder):
    """
    run all parts of the pipeline and write sample images to ouput folder
    :param straight_image_name: image used for picking quad points during perspective initialization
    :param calibration_image_names: list of images to use for camera calibration
    :param road_image_names: list of images to apply pipeline to
    :param output_folder: folder to write output images to
    """
    if not path.exists(output_folder):
        os.makedirs(output_folder)

    # initialization of the pipeline will calibrate camera and init the perspective transforms
    pipeline = RoadAheadPipeline()

    # camera calibration
    pipeline.calibrate_camera(calibration_image_names)

    # perspective transform initialization
    pipeline.initialize_perspective_transform(cv2.imread(straight_image_name))

    for i, file_name in enumerate(road_image_names):
        base_name = path.split(file_name)[1]
        original_image = cv2.imread(file_name)

        # undistort the image
        corrected_image = pipeline.undistort_image(original_image)
        save_single_example('/'.join([output_folder, "corrected_{}".format(base_name)]),
                            "corrected", cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))

        # create threshholded binary image
        binary_image = pipeline.binary_image(corrected_image)
        save_single_example('/'.join([output_folder, "pipeline_binary_{}".format(base_name)]),
                            "binary", binary_image, cmap='gray')

        save_full_example('/'.join([output_folder, "pipeline_binary_process_{}".format(base_name)]),
                          cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                          cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB),
                          binary_image)

        transformed = pipeline.perspective_transform_image(binary_image)
        save_single_example('/'.join([output_folder, "pipeline_transformed_{}".format(base_name)]),
                            "pipeline transformed", transformed, cmap='gray')

        searched_image, left_lane_inds, right_lane_inds = pipeline.first_video_frame_image(transformed)
        fl.visualize_sliding_search('/'.join([output_folder, "pipeline_sliding_search_{}".format(base_name)]),
                                    "sliding search", transformed, searched_image,
                                    pipeline.left_fit, pipeline.right_fit, left_lane_inds, right_lane_inds)

        # fit the same image again since all we did so far is init the histo
        pipeline.fit_frame_image(transformed)

        road_ahead_image = \
            pipeline.road_ahead_image(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB), transformed)
        fl.save_image('/'.join([output_folder, "pipeline_road_ahead_{}".format(base_name)]),
                      "pipeline road ahead", road_ahead_image)


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
    demo_pipeline(calibrate_names, road_names, straight_image, output_folder)


if __name__ == "__main__":
    main()

