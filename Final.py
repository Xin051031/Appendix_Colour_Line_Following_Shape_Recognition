from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import math
import logging

# Motor driver pin definitions
ENA = 18
ENB = 13
IN1 = 17
IN2 = 27
IN3 = 22
IN4 = 23

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)

# Setup motor driver pins
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Setup PWM on the ENA and ENB pins
pwmA = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency for motor 1
pwmB = GPIO.PWM(ENB, 1000)  # 1000 Hz frequency for motor 2
pwmA.start(0)  # Start PWM for motor 1 with 0% duty cycle (off)
pwmB.start(0)  # Start PWM for motor 2 with 0% duty cycle (off)

# Setup camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# Motor movement functions
def move_forward(speed1=50, speed2=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed1)
    pwmB.ChangeDutyCycle(speed2)
    
def move_backward(speed1=50, speed2=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed1)
    pwmB.ChangeDutyCycle(speed2)
    
def move_right(speed1=50, speed2=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed1)
    pwmB.ChangeDutyCycle(speed2)

def move_left(speed1=50, speed2=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed1)
    pwmB.ChangeDutyCycle(speed2)
    
def stop():
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

templates = {
    "square_arrow_forward": cv2.imread("/home/pi/pictures/square_arrow_forward1.jpeg", 0),
    "square_arrow_backward": cv2.imread("/home/pi/pictures/square_arrow_backward1.jpeg", 0),
    "square_arrow_left": cv2.imread("/home/pi/pictures/square_arrow_left1.jpeg", 0),
    "square_arrow_right": cv2.imread("/home/pi/pictures/square_arrow_right1.jpeg", 0),
    "circle_arrow_forward": cv2.imread("/home/pi/pictures/circle_arrow_forward1.jpg", 0),
    "circle_arrow_backward": cv2.imread("/home/pi/pictures/circle_arrow_backward3.jpeg", 0),
    "circle_arrow_left": cv2.imread("/home/pi/pictures/circle_arrow_left1.jpeg", 0),
    "circle_arrow_right": cv2.imread("/home/pi/pictures/circle_arrow_right1.jpeg", 0),
}

color_ranges = {
    "blue": [np.array([100, 80, 40]), np.array([120, 255, 255])],
    "red": [np.array([0, 130, 80]), np.array([8, 255, 255])],  # Tighter red range
    "red": [np.array([0, 100, 100]), np.array([10, 255, 255])],  # Lower red range
    "red2": [np.array([160, 100, 100]), np.array([180, 255, 255])],  # Upper red range
    "green": [np.array([35, 100, 100]), np.array([85, 255, 255])],
    "yellow": [np.array([15, 100, 100]), np.array([35, 255, 255])],
    "black": [np.array([0, 0, 0]), np.array([180, 50, 30])]  # Only very dark pixels
}


# Function to apply color mask
def get_color_mask(hsv, color="black"):
    lower, upper = color_ranges[color]
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def detect_prioritized_color(hsv):
    prioritized_colors = [ "blue",  "yellow", "black"]
    for color in prioritized_colors:
        lower, upper = color_ranges[color]
        mask = cv2.inRange(hsv, lower, upper)
        if cv2.countNonZero(mask) > 900:
            return color, mask
    return None, None

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
# (Previous imports, GPIO setup, motor functions, and color ranges remain unchanged)

# Main loop to track and control motor based on camera input
try:
    previous_color=None
    shape_detected = False
    while True:
        frame = picam2.capture_array()
        height, width = frame.shape[:2]

                # === 1. Arrow Detection (Full Frame) ===
        arrow_detected = False
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for arrow_name, template in templates.items():
            if template is None:
                continue
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > 0.7:  # Adjust threshold if needed
                arrow_detected = True
                direction = direction_map.get(arrow_name, "Unknown")
                cv2.putText(frame, f"Arrow: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)
                print(f"Arrow Detected: {direction}")
                break  # Stop after first detected arrow


        # 1. Shape Detection (Full Frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_detected = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                shape = get_shape_name(cnt, approx)
                if shape != "Unknown":
                    shape_detected = True
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Shape: {shape}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    logging.info(f"Shape detected: {shape}")
                    cv2.putText(frame, shape, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Optional: Stop or perform specific action when shape is detected
                    stop()  # Stop motors when shape is detected (modify as needed)
                    break

        if not shape_detected:

            # Focus only on the lower third of the frame
            crop_y_start = int(height * 1/2)
            roi = frame[crop_y_start:height, 0:width]

            # Convert ROI to HSV for color-based processing
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            detected_color, mask = detect_prioritized_color(hsv_roi)
            print(f"Detected color: {detected_color}")
            
            

            # Perform morphological operations to remove noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the ROI mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate moments to find center
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Adjust cy to full frame coordinates for drawing
                    cy_full = cy + crop_y_start
                    
                    # Get bounding rectangle to analyze shape
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = w / float(h) if h != 0 else 1
                    
                    # Fit a line to the contour to determine orientation
                    [vx, vy, x0, y0] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    angle = math.atan2(vy, vx) * 180 / math.pi  # Convert to degrees
                    
                    # Draw contour and center point on the full frame
                    # Adjust contour coordinates to full frame
                    largest_contour_full = largest_contour + [0, crop_y_start]
                    cv2.drawContours(frame, [largest_contour_full], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy_full), 5, (255, 0, 0), -1)
                    
                    # Decision logic for "n" or "u" shapes
                    if aspect_ratio > 1.5 and abs(angle) > 30:
                        # Handle gentle turn instead of full lock
                        if angle > 0:
                            direction = "Gentle Left (U/N)"
                            move_forward(30, 50)  # Left motor slower
                        else:
                            direction = "Gentle Right (U/N)"
                            move_forward(50, 30)  # Right motor slower

                        
                    if cx < 170:
                        direction = "Hard Left"
                        move_left(65, 65)
                        time.sleep(0.1)
                    elif cx < 240:
                        direction = "Soft Left"
                        move_forward(25, 30)
                    elif cx > 430:
                        direction = "Hard Right"
                        move_right(65, 65)
                        time.sleep(0.1)
                    elif cx > 360:
                        direction = "Soft Right"
                        move_forward(30, 25)
                    else:
                        direction = "Go Straight"
                        move_forward(60, 60)


                    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    print(direction)
                else:
                    print("No Line Detected")
                    stop()
            
        
        # Show processed frame
        cv2.imshow("Track Detection", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped by user.")
finally:
    print("Cleaning up GPIO and closing camera.")
    stop()
    GPIO.cleanup()
    picam2.close()
    cv2.destroyAllWindows()
