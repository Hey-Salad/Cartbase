import cv2
import requests

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

# Capture an image
ret, frame = camera.read()
if ret:
    # Save the image locally
    cv2.imwrite("captured_image.jpg", frame)
    print("Image captured and saved as captured_image.jpg")

    # Send the image to Node-RED via HTTP
    url = "http://localhost:1880/upload-image"  
    files = {"file": open("captured_image.jpg", "rb")}
    response = requests.post(url, files=files)
    print("Image uploaded to Node-RED:", response.text)
else:
    print("Failed to capture image")

# Release the camera
camera.release()