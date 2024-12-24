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

    lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # Debugging: Draw raw lines on a blank image
    debug_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("Hough Transform - Raw Lines", debug_image)
    cv2.waitKey(1)

    return lines


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
                if x1 < width * 0.3:  # Outer left
                    left_outer.append((slope, intercept))
                elif x1 < width * 0.5:  # Inner left
                    left_inner.append((slope, intercept))
            else:  # Positive slope: right-side lanes
                if x1 > width * 0.7:  # Outer right
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

def interpolate_missing_lines(image, categorized_lines):
    """
    Handle missing lines by interpolation or estimation.
    """
    height, width = image.shape[:2]
    interpolated_lines = []
    for lines in categorized_lines:
        avg_line = average_lines(lines)
        if avg_line is None:
            # Default slope and intercept if missing
            slope = -0.5 if len(interpolated_lines) < 2 else 0.5
            intercept = height if slope < 0 else 0
            avg_line = (slope, intercept)
        interpolated_lines.append(avg_line)
    return interpolated_lines


def pixel_coordinates(y1, y2, line):
    """
    Convert slope-intercept form to pixel coordinates.
    Ensure coordinates are within the frame boundaries.
    """
    if line is None:
        return None
    slope, intercept = line
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    except ZeroDivisionError:
        return None
    # Ensure coordinates are within image bounds
    return (max(0, min(x1, 1920)), y1), (max(0, min(x2, 1920)), y2)


def lane_lines(image, lines):
    """
    Generate four lane lines from detected lines.
    """
    height, width = image.shape[:2]
    y1 = height
    y2 = int(height * 0.6)

    categorized_lines = categorize_lines(lines, width)
    interpolated_lines = interpolate_missing_lines(image, categorized_lines)

    return [
        pixel_coordinates(y1, y2, interpolated_lines[0]),  # Left outer
        pixel_coordinates(y1, y2, interpolated_lines[1]),  # Left inner
        pixel_coordinates(y1, y2, interpolated_lines[2]),  # Right inner
        pixel_coordinates(y1, y2, interpolated_lines[3]),  # Right outer
    ]

def debug_frame(image, lines):
    """
    Visualize the frame with detected lanes before combining into the final video.
    """
    debug_image = draw_lane_lines(image, lines)
    cv2.imshow("Debug Frame with Lanes", debug_image)
    cv2.waitKey(1)  # Adjust to 1 for real-time debugging
    return debug_image

def draw_lane_lines(image, lines):
    """
    Draw the detected lane lines on the image.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, line[0], line[1], (0, 255, 0), 10)  # Green color with thickness 10
    # Overlay the lines on the original image
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def frame_processor(image):
    """
    Process each frame to detect and draw lane lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_selection(edges)
    lines = hough_transform(roi)

    if lines is not None:
        lanes = lane_lines(image, lines)
        debug_frame(image, lanes)  # Optional: Debug visualization
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
