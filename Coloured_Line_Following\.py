from picamera2 import Picamera2
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import math

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

color_ranges = {
    "blue": [np.array([100, 80, 40]), np.array([120, 255, 255])],
    "red": [np.array([0, 130, 80]), np.array([10, 255, 255])],  # Tighter red range
    "green": [np.array([35, 100, 100]), np.array([85, 255, 255])],
    "yellow": [np.array([15, 100, 100]), np.array([35, 255, 255])],
    "black": [np.array([0, 0, 0]), np.array([180, 100, 70])]  # Only very dark pixels
}

# Function to apply color mask
def get_color_mask(hsv, color="black"):
    lower, upper = color_ranges[color]
    mask = cv2.inRange(hsv, lower, upper)
    return mask

# Function to detect the prioritized color
def detect_prioritized_color(hsv):
    priority_colors = ["yellow","blue","black"]
    
    for color in priority_colors:
        mask = get_color_mask(hsv, color)
        if cv2.countNonZero(mask) > 900:
            return color, mask
    return None, mask

# (Previous imports, GPIO setup, motor functions, and color ranges remain unchanged)

# Main loop to track and control motor based on camera input
try:
    previous_color=None
    while True:
        frame = picam2.capture_array()
        height, width = frame.shape[:2]

        # Focus only on the lower half of the frame
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
                
                # Fallback to centroid-based decision for straight lines
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
                print(f"Aspect Ratio: {aspect_ratio:.2f}, Angle: {angle:.2f} degrees")
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

