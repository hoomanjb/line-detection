import numpy as np
import cv2
from moviepy.editor import VideoFileClip


def region_selection(image):
    """
    Mask the region of interest (ROI) to focus on the road lanes.
    """
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        polygon_color = (0, 255, 0)  # Green for visualization
    else:
        ignore_mask_color = 255
        polygon_color = 255  # White for single channel

    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.05, rows * 0.9]  # Adjusted for wider bottom
    top_left = [cols * 0.25, rows * 0.45]  # Adjusted for wider top
    bottom_right = [cols * 0.95, rows * 0.9]  # Adjusted for wider bottom
    top_right = [cols * 0.75, rows * 0.45]  # Adjusted for wider top

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(image):
    """
    Apply Hough Transform to detect lines.
    """
    rho = 1  # Reduced for more precision
    theta = np.pi / 180
    threshold = 30  # Increased threshold for more prominent lines
    minLineLength = 20  # Reduced to capture shorter lines
    maxLineGap = 20  # Reduced to connect close line segments
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)


def categorize_lines(lines, width):
    """
    Split lines into four categories: left outer, left inner, right inner, right outer.
    """
    left_outer, left_inner, right_inner, right_outer = [], [], [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue  # Skip vertical lines
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if slope < -0.2:  # Left-side lanes, adjusted slope threshold
                if x1 < width * 0.45:  # Adjusted outer threshold
                    left_outer.append((slope, intercept))
                else:  # Inner left
                    left_inner.append((slope, intercept))
            elif slope > 0.2:  # Right-side lanes, adjusted slope threshold
                if x1 > width * 0.55:  # Adjusted outer threshold
                    right_outer.append((slope, intercept))
                else:  # Inner right
                    right_inner.append((slope, intercept))

    return left_outer, left_inner, right_inner, right_outer


def average_lines(lines):
    """
    Average lines to generate a single representative line.
    """
    if len(lines) == 0:
        return None
    slope, intercept = np.mean(lines, axis=0)
    return slope, intercept


def pixel_coordinates(y1, y2, line):
    """
    Convert slope-intercept form to pixel coordinates.
    """
    if line is None:
        return None
    slope, intercept = line

    # Avoid division by zero in cases where slope is close to zero
    if abs(slope) < 1e-6:  # Small threshold to consider slope as zero
        # Handle nearly horizontal lines by setting x to an arbitrary large value
        x1 = int(1e6)  # A large value
        x2 = int(-1e6)  # A large negative value
        y1 = int(intercept)  # y-coordinate remains the same
        y2 = int(intercept)  # y-coordinate remains the same
    else:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

    return (x1, y1), (x2, y2)


def lane_lines(image, lines):
    """
    Generate four lane lines from detected lines.
    """
    height, width = image.shape[:2]
    y1 = height  # Bottom of the image
    y2 = int(height * 0.55)  # Adjusted y2 to be slightly higher

    left_outer, left_inner, right_inner, right_outer = categorize_lines(lines, width)

    left_outer_avg = average_lines(left_outer)
    left_inner_avg = average_lines(left_inner)
    right_inner_avg = average_lines(right_inner)
    right_outer_avg = average_lines(right_outer)

    return [
        pixel_coordinates(y1, y2, left_outer_avg),
        pixel_coordinates(y1, y2, left_inner_avg),
        pixel_coordinates(y1, y2, right_inner_avg),
        pixel_coordinates(y1, y2, right_outer_avg),
    ]


def draw_lane_lines(image, lines, color=(0, 255, 0), thickness=10):
    """
    Draw the detected lane lines on the image.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def frame_processor(image):
    """
    Process each frame to detect lane lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_selection(edges)
    lines = hough_transform(roi)
    if lines is not None:
        lanes = lane_lines(image, lines)
        return draw_lane_lines(image, lanes)
    return image


def process_video(input_path, output_path):
    """
    Process the video to detect lane lines.
    """
    video = VideoFileClip(input_path)
    processed = video.fl_image(frame_processor)
    processed.write_videofile(output_path, audio=False)


# Run the function
process_video("17.MOV", "output2.mp4")