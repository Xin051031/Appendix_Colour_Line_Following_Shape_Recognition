import cv2
import numpy as np
from picamera2 import Picamera2
import math
import time
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load all templates
templates = {
    "stop_circle": cv2.imread("/home/pi/pictures/stop_circle.jpeg", 0),
    "stop_hand": cv2.imread("/home/pi/pictures/stop_hand.jpeg", 0),
    "face_recognition": cv2.imread("/home/pi/pictures/face_recognize.jpeg", 0),
    "distance": cv2.imread("/home/pi/pictures/distance_calculation.jpeg", 0),
    "square_arrow_forward": cv2.imread("/home/pi/pictures/square_arrow_forward1.jpeg", 0),
    "square_arrow_backward": cv2.imread("/home/pi/pictures/square_arrow_backward1.jpeg", 0),
    "square_arrow_left": cv2.imread("/home/pi/pictures/square_arrow_left1.jpeg", 0),
    "square_arrow_right": cv2.imread("/home/pi/pictures/square_arrow_right1.jpeg", 0),
    "circle_arrow_forward": cv2.imread("/home/pi/pictures/circle_arrow_forward1.jpg", 0),
    "circle_arrow_backward": cv2.imread("/home/pi/pictures/circle_arrow_backward3.jpeg", 0),
    "circle_arrow_left": cv2.imread("/home/pi/pictures/circle_arrow_left1.jpeg", 0),
    "circle_arrow_right": cv2.imread("/home/pi/pictures/circle_arrow_right1.jpeg", 0),
}

# Validate template loading
for label, img in templates.items():
    if img is None:
        logging.error(f"Could not load template for {label}")
        exit()

# ORB + FLANN init
#initialize ORB detector
orb = cv2.ORB_create(nfeatures=750)
#Define FLANN Parameter and create matchers
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Precompute ORB descriptors for templates
template_features = {}
for label, img in templates.items():
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        logging.warning(f"No descriptors found for {label}")
    template_features[label] = (kp, des)

# Camera setup
try:
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "BGR888"
    picam2.configure("preview")
    picam2.start()
except Exception as e:
    logging.error(f"Failed to initialize camera: {e}")
    exit(1)

def get_shape_name(contour, approx):
    num_sides = len(approx)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "Unknown"
    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    
    if num_sides == 3:
        return "Triangle"
    elif num_sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        return "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif num_sides == 5:
        return "Pentagon"
    elif num_sides == 6:
        return "Hexagon"
    elif circularity > 0.85 and area > 500:
        return "Circle"
    elif 0.6 <= circularity <= 0.85 and area > 300:
        # Check for a gap using convexity defects (with error handling)
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None and defects.shape[0] > 0 and defects.shape[2] == 4:  # Ensure valid shape
                max_defect = max(defects[:, 0, :], key=lambda x: x[3])  # Extract defect row
                if max_defect[3] / 256 > 5:  # Distance value at index 3
                    return "Three-Quarter Circle"
        except Exception as e:
            logging.debug(f"Defect calculation failed: {e}")
        # Fallback to circularity if defect check fails
        return "Three-Quarter Circle"
    return "Unknown"

# Arrow direction map
direction_map = {
    "square_arrow_forward": "Forward",
    "square_arrow_backward": "Backward",
    "square_arrow_left": "Left",
    "square_arrow_right": "Right",
    "circle_arrow_forward": "Forward",
    "circle_arrow_backward": "Backward",
    "circle_arrow_left": "Left",
    "circle_arrow_right": "Right"
}

# History buffers
symbol_history = []
arrow_history = []
history_size = 5

# Processing loop
try:
    prev_time = time.time()
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 100)

        arrow_detected = False
        symbol_detected = False

        # ORB on current frame
        kp_frame, des_frame = orb.detectAndCompute(gray, None)

        if des_frame is None or len(des_frame) < 2:
            logging.warning("Too few descriptors in the frame. Skipping matching.")
        else:
            best_arrow_label = None
            best_arrow_matches = 0
            best_symbol_label = None
            best_symbol_matches = 0

            for label, (kp_template, des_template) in template_features.items():
                if des_template is None or len(des_template) < 2:
                    continue

                try:
                    #Match Descriptors
                    matches = flann.knnMatch(des_template, des_frame, k=2)
                except cv2.error as e:
                    logging.warning(f"FLANN error for {label}: {e}")
                    continue

                good_matches = [m for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

                if "arrow" in label:
                    if len(good_matches) > best_arrow_matches:
                        best_arrow_matches = len(good_matches)
                        best_arrow_label = label
                else:
                    if len(good_matches) > best_symbol_matches:
                        best_symbol_matches = len(good_matches)
                        best_symbol_label = label
                        


            # Arrow detection result
            if best_arrow_label and best_arrow_matches > 15:
                arrow_detected = True
                direction = direction_map.get(best_arrow_label, "Unknown")
                arrow_history.append(direction)
                if len(arrow_history) > history_size:
                    arrow_history.pop(0)
                smoothed_arrow = Counter(arrow_history).most_common(1)[0][0]
                cv2.putText(frame_bgr, f"Arrow: {smoothed_arrow}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                logging.info(f"ORB Arrow: {smoothed_arrow}")

            # Symbol detection result
            if not arrow_detected and best_symbol_label and best_symbol_matches > 18:
                symbol_detected = True
                symbol_history.append(best_symbol_label)
                if len(symbol_history) > history_size:
                    symbol_history.pop(0)
                smoothed_symbol = Counter(symbol_history).most_common(1)[0][0]
                cv2.putText(frame_bgr, f"Symbol: {smoothed_symbol}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                logging.info(f"ORB Symbol: {smoothed_symbol}")

        # Shape detection fallback
        if not arrow_detected and not symbol_detected:
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    shape = get_shape_name(cnt, approx)
                    if shape != "Unknown":
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.drawContours(frame_bgr, [cnt], -1, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, f"Shape: {shape}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        logging.info(f"Shape detected: {shape}")
                        break

        # FPS display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the result
        cv2.imshow("Detection", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logging.info("Program stopped by user")
except Exception as e:
    logging.error(f"Error occurred: {e}")
finally:
    picam2.stop()
    cv2.destroyAllWindows()

