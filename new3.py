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
    Apply Hough Transform to detect lines.
    """
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 50
    maxLineGap = 200
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

            if slope < 0:  # Negative slope: left-side lanes
                if x1 < width * 0.25:  # Outer left
                    left_outer.append((slope, intercept))
                elif x1 < width * 0.5:  # Inner left
                    left_inner.append((slope, intercept))
            else:  # Positive slope: right-side lanes
                if x1 > width * 0.75:  # Outer right
                    right_outer.append((slope, intercept))
                elif x1 > width * 0.5:  # Inner right
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
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except ZeroDivisionError:
        return None
    return (x1, y1), (x2, y2)


def lane_lines(image, lines):
    """
    Generate four lane lines from detected lines.
    """
    height, width = image.shape[:2]
    y1 = height
    y2 = int(height * 0.6)

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


def draw_lane_lines(image, lines):
    """
    Draw the detected lane lines on the image.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, (0, 255, 0), 10)
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
process_video("17.MOV", "output.mp4")
