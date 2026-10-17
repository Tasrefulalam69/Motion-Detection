import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Read the first two consecutive frames to compute motion difference
ret, frame1 = cap.read()
ret, frame2 = cap.read()

print("Project 2: Motion Detection Started... Press 'q' to exit.")

while cap.isOpened():
    if not ret:
        break

    # 1. Compute absolute difference between consecutive frames
    diff = cv2.absdiff(frame1, frame2)
    
    # 2. Convert the difference frame to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Gaussian Blur to reduce high-frequency noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Apply thresholding to isolate moving regions
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # 5. Dilate the thresholded image to fill in gaps/holes
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # 6. Find contours of the moving objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        # Ignore small movements/noise by checking contour area threshold
        if cv2.contourArea(contour) < 2000:
            continue
        
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Draw bounding rectangle around the detected motion
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Display status alert text on screen
    if motion_detected:
        cv2.putText(frame1, "STATUS: MOTION DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame1, "STATUS: CLEAR", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Flip image horizontally for a natural mirror perspective
    display_frame = cv2.flip(frame1, 1)

    # Display video output window
    cv2.imshow("Project 2 - Motion Detection System", display_frame)

    # Update frames for the next loop iteration
    frame1 = frame2
    ret, frame2 = cap.read()

    # Press 'q' on the keyboard to exit the application
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()