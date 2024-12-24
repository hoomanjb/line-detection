import numpy as np
import cv2
from moviepy.editor import VideoFileClip


def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
    """
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]

    # Define vertices for a trapezoid that encompasses more of the road for 4-line detection
    bottom_left = [cols * 0.001, rows * 0.99]
    middle_left = [cols * 0.4, rows * 0.65]
    middle_right = [cols * 0.6, rows * 0.65]
    bottom_right = [cols * 0.99, rows * 0.99]

    # Vertices for the inner lines (assuming they are closer to the center)
    inner_left_top = [cols * 0.35, rows * 0.65]
    inner_left_bottom = [cols * 0.25, rows * 0.99]
    inner_right_top = [cols * 0.65, rows * 0.65]
    inner_right_bottom = [cols * 0.75, rows * 0.99]

    # Combine all vertices to form two separate polygons
    vertices_outer = np.array([[bottom_left, middle_left, middle_right, bottom_right]], dtype=np.int32)
    vertices_inner = np.array([[inner_left_top, inner_left_bottom, inner_right_bottom, inner_right_top]],
                              dtype=np.int32)

    # Filling the two polygons with white color and generating the final mask
    cv2.fillPoly(mask, vertices_outer, ignore_mask_color)
    cv2.fillPoly(mask, vertices_inner, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def hough_transform(image):
    """
    Apply Hough Transform to detect lines.
    """
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 25
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def average_slope_intercept(lines, image_shape):
    """
    Calculate the average slope and intercept for left, right, inner-left, and inner-right lines.
    """
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    inner_left_lines = []
    inner_left_weights = []
    inner_right_lines = []
    inner_right_weights = []

    image_width = image_shape[1]

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

            # Classify lines based on slope and position
            if slope < 0:  # Left side lines
                if x1 < image_width * 0.4 and x2 < image_width * 0.4:  # Outer left
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                elif x1 < image_width * 0.5 and x2 < image_width * 0.5:  # Inner left
                    inner_left_lines.append((slope, intercept))
                    inner_left_weights.append(length)
            else:  # Right side lines
                if x1 > image_width * 0.6 and x2 > image_width * 0.6:  # Outer right
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
                elif x1 > image_width * 0.5 and x2 > image_width * 0.5:  # Inner right
                    inner_right_lines.append((slope, intercept))
                    inner_right_weights.append(length)

    # Calculate average slope and intercept for each line group
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    inner_left_lane = np.dot(inner_left_weights, inner_left_lines) / np.sum(inner_left_weights) if len(
        inner_left_weights) > 0 else None
    inner_right_lane = np.dot(inner_right_weights, inner_right_lines) / np.sum(inner_right_weights) if len(
        inner_right_weights) > 0 else None

    return left_lane, right_lane, inner_left_lane, inner_right_lane


def pixel_points(y1, y2, line):
    """
    Convert the slope and intercept of a line into pixel points.
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
    Create full-length lines from pixel points for each lane.
    """
    # Pass image.shape to average_slope_intercept
    left_lane, right_lane, inner_left_lane, inner_right_lane = average_slope_intercept(lines, image.shape)
    y1 = image.shape[0]
    y2 = y1 * 0.65

    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    inner_left_line = pixel_points(y1, y2, inner_left_lane)
    inner_right_line = pixel_points(y1, y2, inner_right_lane)

    return left_line, right_line, inner_left_line, inner_right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image, now with 4 lines.
    """
    line_image = np.zeros_like(image)

    # Define different colors for inner and outer lines
    outer_line_color = [255, 0, 0]  # Red for outer lines
    inner_line_color = [0, 0, 255]  # Blue for inner lines

    if lines is not None:
        left_line, right_line, inner_left_line, inner_right_line = lines

        # Draw the lines with corresponding colors
        if left_line is not None:
            cv2.line(line_image, *left_line, outer_line_color, thickness)
        if right_line is not None:
            cv2.line(line_image, *right_line, outer_line_color, thickness)
        if inner_left_line is not None:
            cv2.line(line_image, *inner_left_line, inner_line_color, thickness)
        if inner_right_line is not None:
            cv2.line(line_image, *inner_right_line, inner_line_color, thickness)

    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def frame_processor(image):
    """
    Process the input frame to detect lane lines.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    region = region_selection(edges)
    hough = hough_transform(region)

    # Pass the original image to lane_lines for correct calculations
    lines = lane_lines(image, hough)

    result = draw_lane_lines(image, lines)
    return result


def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
    """
    input_video = VideoFileClip(test_video, audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_video, audio=False)


# Calling driver function
process_video('17.MOV', 'output2.mp4')