from picamera2 import Picamera2
import cv2
import numpy as np

# Global variable for clicked HSV values
clicked_hsv = None
"""Callback function to get HSV values on mouse click."""
def mouse_callback(event, x, y, flags, param):
    global clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param
        clicked_hsv = hsv_frame[y, x]
        print(f"Clicked HSV: H={clicked_hsv[0]}, S={clicked_hsv[1]}, V={clicked_hsv[2]}")

def test_blue_detection(frame):
    print("Processing frame...")
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    print("Converted to HSV")
    
    # Split HSV into separate channels
    hue, saturation, value = cv2.split(hsv)
    print("Split HSV channels")
    
    # Very broad initial range to catch any blue-like color
    lower_blue = np.array([20, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Create mask for blue
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    print("Created blue mask")
    
    # Show the original frame first and set callback
    cv2.imshow("Original Frame", frame)
    cv2.setMouseCallback("Original Frame", mouse_callback, hsv)
    print("Displayed Original Frame and set callback")
    
    # Show HSV channels
    cv2.imshow("Hue Channel", hue)
    cv2.imshow("Saturation Channel", saturation)
    cv2.imshow("Value Channel", value)
    print("Displayed HSV channels")
    
    # Show the blue mask
    cv2.imshow("Blue Mask", blue_mask)
    print("Displayed Blue Mask")
    
    # Show the original frame with blue areas highlighted
    blue_highlight = cv2.bitwise_and(frame, frame, mask=blue_mask)
    cv2.imshow("Blue Highlight", blue_highlight)
    print("Displayed Blue Highlight")
    
    # Display clicked HSV value on frame
    if clicked_hsv is not None:
        text = f"H={clicked_hsv[0]}, S={clicked_hsv[1]}, V={clicked_hsv[2]}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

try:
    picam2.start()
    print("Camera started successfully")
    
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print("Captured frame")
        
        processed_frame = test_blue_detection(frame)
        # Update Original Frame with text
        cv2.imshow("Original Frame", processed_frame)
        print("Updated Original Frame")
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"Error: {str(e)}")
    
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped and windows closed")
