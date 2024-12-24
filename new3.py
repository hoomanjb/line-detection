import numpy as np
import cv2

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

import imageio


def visualize_roi(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.001, rows * 0.99]
    top_left = [cols * 0.001, rows * 0.4]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.9, rows * 0.4]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    overlay_image = image.copy()
    cv2.polylines(overlay_image, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Region of Interest", overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: we pass here the output from canny where we have
        identified edges in the frame
    """
    # create an array of the same size as of the input image
    mask = np.zeros_like(image)
    # if you pass an image with more then one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        polygon_color = (0, 255, 0)  # Green for visualization
    else:
        ignore_mask_color = 255
        polygon_color = 255  # White for single channel

    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed

    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.001, rows * 0.85]
    top_left = [cols * 0.25, rows * 0.2]

    bottom_right = [cols * 0.45, rows * 0.99]
    top_right = [cols * 0.4, rows * 0.2]


    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road

    # overlay_image = image.copy()
    # cv2.polylines(overlay_image, [vertices], isClosed=True, color=polygon_color, thickness=2)
    # cv2.imshow("Polygon Overlay", overlay_image)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
    Parameter:
        image: grayscale image which should be an output from the edge detector
    """
    # Distance resolution of the accumulator in pixels.
    rho = 1
    # Angle resolution of the accumulator in radians.
    theta = np.pi / 180
    # Only lines that are greater than threshold will be returned.
    threshold = 20
    # Line segments shorter than that are rejected.
    minLineLength = 20
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500
    # function returns an array containing dimensions of straight lines
    # appearing in the input image
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def average_slope_intercept(lines, image_width):
    """
    Categorize lines into left-inner, left-outer, right-inner, and right-outer.
    Parameters:
        lines: Output from Hough Transform
        image_width: Width of the image to differentiate inner and outer lines
    """
    left_outer_lines = []  # (slope, intercept)
    left_inner_lines = []  # (slope, intercept)
    right_outer_lines = []  # (slope, intercept)
    right_inner_lines = []  # (slope, intercept)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Ignore vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Categorize lines based on slope and position
            if slope < 0:  # Left side
                if x1 < image_width * 0.5 and x2 < image_width * 0.5:
                    if len(left_inner_lines) == 0 or slope < min([l[0] for l in left_inner_lines]):
                        left_inner_lines.append((slope, intercept))
                    else:
                        left_outer_lines.append((slope, intercept))
            elif slope > 0:  # Right side
                if x1 > image_width * 0.5 and x2 > image_width * 0.5:
                    if len(right_inner_lines) == 0 or slope > max([l[0] for l in right_inner_lines]):
                        right_inner_lines.append((slope, intercept))
                    else:
                        right_outer_lines.append((slope, intercept))

    def average_lines(lines):
        if len(lines) == 0:
            return None
        slope, intercept = np.mean(lines, axis=0)
        return slope, intercept

    return (
        average_lines(left_outer_lines),
        average_lines(left_inner_lines),
        average_lines(right_outer_lines),
        average_lines(right_inner_lines),
    )

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
    Parameters:
        y1: y-value of the line's starting point.
        y2: y-value of the line's end point.
        line: The slope and intercept of the line.
    Returns:
        A tuple containing the starting and ending points of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    """
    Generate pixel points for four lanes.
    """
    image_width = image.shape[1]
    left_outer_lane, left_inner_lane, right_outer_lane, right_inner_lane = average_slope_intercept(lines, image_width)
    y1 = image.shape[0]
    y2 = int(y1 * 0.4)

    left_outer_line = pixel_points(y1, y2, left_outer_lane)
    left_inner_line = pixel_points(y1, y2, left_inner_lane)
    right_outer_line = pixel_points(y1, y2, right_outer_lane)
    right_inner_line = pixel_points(y1, y2, right_inner_lane)

    return [left_outer_line, left_inner_line, right_outer_line, right_inner_line]


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    """
    Draw detected lane lines on the image.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def frame_processor(image):
    """
    Process the input frame to detect four lane lines.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    hough = hough_transform(region)
    if hough is not None:
        lanes = lane_lines(image, hough)
        result = draw_lane_lines(image, lanes)
    else:
        result = image
    return result

# driver function
def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
    Parameters:
        test_video: location of input video file
        output_video: location where output video file is to be saved
    """
    # read the video file using VideoFileClip without audio
    input_video = VideoFileClip(test_video, audio=False)
    # apply the function "frame_processor" to each frame of the video
    # will give more detail about "frame_processor" in further steps
    # "processed" stores the output video
    processed = input_video.fl_image(frame_processor)
    # save the output video stream to an mp4 file
    processed.write_videofile(output_video, audio=False)


# calling driver function
process_video('17.MOV', 'output.mp4')