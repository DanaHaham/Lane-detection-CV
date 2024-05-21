import cv2
import matplotlib.pyplot as plt
import numpy as np

# Open video in the given path
def open_video(video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully loaded
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    return cap

# Release the given videos
def release_videos(video_input, video_output):
    video_input.release()
    video_output.release()

# Create new video in the given path acordding to the given properties
def create_video(output_path, fps, frame_width, frame_height):

    # Create the video file
    write = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    return write 

# Draw the given line in the given image
def draw_line(image, line):

    if line is not None:
        y1 = image.shape[0]
        x1 = int((y1 - line[1]) / line[0])
        y2 = int(image.shape[0] * 0.6)
        x2 = int((y2 - line[1]) / line[0])
        cv2.line(image, (x1, y1), (x2, y2), (118, 68, 255), thickness= 5)
    
    return image

# Draw the given direction in the given image
def draw_dir(image, dir):

    if dir == 'f':
        return image
    
    elif dir == 'r':
        turn = cv2.imread('data/right_turn.png')
        msg = "Right Turn Ahead"

    elif dir == 'l':
        turn = cv2.imread('data/left_turn.png')
        msg = "Left Turn Ahead"

    image[100:100+turn.shape[0], 100:turn.shape[1]+100] = turn
    cv2.putText(image, msg, org=(70, 350), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    return image

# Draw the given rectangle in the given image
def draw_rec(image, rec):
    if rec is not None:
        x1, y1, x2, y2 = rec

        # Mark how close is the car 
        if y2 > image.shape[0] / 1.45:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        # Draw a rectangle around the detected region
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    return image

# Crop the given image without losing essential information
def crop_image(image, index_video):

    # Init the roi of each vide0
    roi_vertices= []
    
    # Get the height and width of the given image
    height, width = image.shape[:2]

    # Define the region of interest coordinates
    # Adjust these coordinates based on the layout of the road and distractions in the given image
    roi_vertices.append(np.array([[(width, height), (width//6, height), (width // 2.4, height // 1.67), (width // 1.85, height // 1.67)]], dtype=np.int32))
    roi_vertices.append(np.array([[(width, height//1.45), (width//6.5, height//1.45), (width // 2.7, height // 1.67), (width // 1.7, height // 1.67)]], dtype=np.int32))

    # Create a mask of zeros with the same shape as the given image
    mask = np.zeros_like(image)
    
    # Fill the region of interest (ROI) with white color (255)
    cv2.fillPoly(mask, roi_vertices[index_video], (255, 255, 255))
    
    # Apply the mask to the given image
    cropped_image = cv2.bitwise_and(image, mask)

    return cropped_image

# Improve the given night mode image
def improve_low_light(image):
    
    # Increase brightness and contrast
    alpha = -0.3  # Brightness control (1.0-3.0)
    beta = 45  # Contrast control (0-100)
    enhanced_scale = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    enhanced_image = np.zeros_like(image) 
    enhanced_image[enhanced_scale < 15 ] = 255

    return enhanced_image

# Detect the lane and extract their edges
def find_lane_edges(image, index_video):
        
    # Convert image to grayscale
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Clean noises 
    gaussian_im = cv2.GaussianBlur(gray_im,(5, 5), 0)

    # Improve image in case of night
    if index_video == 1:
        gaussian_im = improve_low_light(gaussian_im)

    # Apply threshold - keep only the white lanes
    _, thresh_im = cv2.threshold(gaussian_im, 180, 255, cv2.THRESH_BINARY)

    # Crop unessential information
    cropped_image = crop_image(thresh_im, index_video)

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Perform dilation to fill small gaps in edges
    dilated_edges = cv2.dilate(cropped_image, kernel, iterations=2)

    # Extract edges
    canny_im = cv2.Canny(dilated_edges, 50, 90)
         
    return canny_im

# Verife the left and right lines in the given lines on the given image
def find_left_right_lines(image, lines, index_video):

    # Init
    tresh_center = [310, 100]
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Avoid invalid slope
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)

                # Remove lines that are not lanes acording to thier geometry
                if(abs(slope) > 0.5 and abs(slope) < 1.4):

                    # Calculate the midpoint of the line
                    midpoint_x = (x1 + x2) / 2

                    # Classify the line as left or right based on the midpoint position and distance from the center
                    if abs(midpoint_x - image.shape[1] / 2) < tresh_center[index_video]:
                        continue  # Skip lines too close to the center

                    # Classify the line as left or right based on the midpoint position
                    if midpoint_x < image.shape[1] / 2:
                        left_lines.append((slope, intercept))

                    else:
                        right_lines.append((slope, intercept))
   
    return left_lines, right_lines

# Find the average line of the given lines
def find_avg_line(lines, prev_line, prev_count, index_video):
    
    # Init
    tresh_prev = [40, 9]

    # There is lines to calculate average
    if len(lines) > 0:

        # Calculate the average slope and intercept for the lane
        slopes = [line[0] for line in lines]
        intercepts = [line[1] for line in lines]

        mean_slope = np.mean(slopes)
        mean_intercept = np.mean(intercepts)

        # Combine the prev lane and the current
        if prev_line is not None:
            mean_slope = ((0.15 * mean_slope) + (0.85 * prev_line[0]))
            mean_intercept = ((0.15 * mean_intercept) + (0.85 * prev_line[1]))

        line = ((mean_slope, mean_intercept), 0)

    # There is no line in the current image, check if its annomaly or lane that disappear
    else:
        prev_count = prev_count + 1

        if prev_count < tresh_prev[index_video] and prev_line is not None:
            mean_slope = prev_line[0]
            mean_intercept = prev_line[1]
            line = ((mean_slope, mean_intercept), prev_count)

        else:
            line = (None, prev_count)

    return line

# Find the lanes in the given image   
def find_lanes(image, prev_lanes, index_video):
    
    r_step = 1
    t_step = np.pi / 180
    TH = 20
    
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(image, r_step, t_step, TH, minLineLength= 50, maxLineGap= 100)

    # Find left and right lines
    left_lines , right_lines = find_left_right_lines(image, lines, index_video)

    # Extract the lanes
    left_lane = find_avg_line(left_lines, prev_lanes[0][0], prev_lanes[0][1], index_video)
    right_lane = find_avg_line(right_lines, prev_lanes[1][0], prev_lanes[1][1], index_video)

    return left_lane, right_lane

# Check whether there is lane departure and return the direction - r,l,f
def find_lane_departure(current_left, current_right, index_video):
    
    # Init
    count_none = [40, 9]

    # Left lane disappear before right
    if current_left[1] > count_none[index_video] and current_left[1] - current_right[1] > 0:
        return 'l' 
    
    # Right lane disappear before left
    elif current_right[1] > count_none[index_video] and current_left[1] - current_right[1] < 0:
        return 'r' 
    
    return 'f'

# Find the car acording to the given templates in the frame
def find_car(image, mark_cars, prev_rec, index_temp):
    
    car_range = [17, 14, 11]
    rec = None

    for index_car in range(car_range[index_temp]):
        
        # Load template image
        template = cv2.imread(f'data/cars/{index_temp + 1}/car_{index_car + 1}.jpg', cv2.IMREAD_GRAYSCALE)

        # Load target image
        target = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        match_im = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)

        # Extract the location of the best match
        _, max_val, _, max_loc = cv2.minMaxLoc(match_im)

        # Threshold the result to find potential car locations
        threshold = 0.8

        if max_val > threshold:
               
            # Determine bounding box corners for the match
            h, w = template.shape[:2]
            x1, y1 = max_loc
            x2, y2 = (x1 + w, y1 + h)

            rec = (x1, y1, x2, y2)
            break

    # In case that dosent found
    if rec is None and prev_rec is not None:
        rec = prev_rec

    # Draw rectangle
    mark_cars = draw_rec(mark_cars, rec)

    return mark_cars, rec

# Detect lane in the given image and return them in the given mask_lanes
def detect_lanes(image, mask_lanes, prev_lanes, index_video):

    # Extract edges of the lanes in the given image
    edges_im = find_lane_edges(image, index_video)

    # Find the lanes
    left_lane, right_lane = find_lanes(edges_im, prev_lanes, index_video)
    
    # Draw the lanes
    mask_lanes = draw_line(mask_lanes, left_lane[0])
    mask_lanes = draw_line(mask_lanes, right_lane[0])
        
    return mask_lanes, (left_lane, right_lane)

# Detect lane departure in the given image and return the image with mark
def detect_lane_departure(image, current_lanes, index_video):
    
    # Direction of the car
    dir = find_lane_departure(current_lanes[0], current_lanes[1], index_video)
    image = draw_dir(image, dir)

    return image

# Detect cars in the given image and return them in the given mark_cars
def detect_cars(image, mark_cars, prev_rec):
   
   #Init
    recs = []
   
   # Find all the cars
    for index_car in range(3):
      mark_cars, rec = find_car(image, mark_cars, prev_rec[index_car], index_car)
      recs.append(rec)

    return mark_cars, (recs[0], recs[1], recs[2])
    
# Run the algorithm on the given video   
def handle_video(index_video):

    # Open the video file
    og_video = open_video(f'data/dashcam_{index_video + 1}.mp4')
   
    # Read first frame
    ret, frame = og_video.read()

    # Define the output video writer
    detected_video = create_video(f'output/output_{index_video + 1}.mp4', og_video.get(cv2.CAP_PROP_FPS), int(og_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(og_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Init prev lanes
    prev_lanes = ((None, 0), (None, 0))

    # Init prev rectangle 
    prev_rec = (None, None, None)

    # Init details index
    index_video_detail = index_video

    # Capture farme by frame until the video is over
    while(ret):

        # Create new image for the detected object in the frame 
        detected_frame = np.zeros_like(frame)

        # Detect the cars in the 3td video
        if index_video == 2:
            detected_frame, prev_rec = detect_cars(frame, detected_frame, prev_rec)
            index_video_detail = 0

        # Detect the lanes in the frame
        detected_frame, current_lanes = detect_lanes(frame, detected_frame, prev_lanes, index_video_detail)

        # Blend the lanes in the frame
        res_image = cv2.addWeighted(frame, 0.8 ,detected_frame, 1, 0)

        # Detect lane departure and draw them
        res_image = detect_lane_departure(res_image, current_lanes, index_video_detail)
                
        # Add the detected frame to the output video
        detected_video.write(res_image)
        
        # Continue to the next frame
        ret, frame = og_video.read()
        prev_lanes = current_lanes

    # Release the video
    release_videos(og_video, detected_video)
   
# Main
if __name__ == "__main__":

    # Active the actions on every video
    for i in range (3):
        handle_video(i)
    
    # Close program
    cv2.destroyAllWindows()