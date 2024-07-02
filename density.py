import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture('cv.mp4')  # Replace 'video.mp4' with the path to your video file

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Threshold the frame to obtain a binary mask of the crowd
    _, binary_mask = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the density of people using the number of contours
    density = len(contours)

    # Print the density of people in the crowd
    print(f'Density of people in the crowd: {density}')

    # Display the binary mask
    cv2.imshow('Binary Mask', binary_mask)

    # Press 'q' to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
