import cv2
import numpy as np
# Load the image
img = cv2.imread('friendscast.jpg')
# Check if image loaded successfully
if img is None:
    print("Error loading image")
    exit()
# **1. Resizing**
resized_img = cv2.resize(img, (300, 200))  # Resize to width=300, height=200
# **2. Grayscale conversion**
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# **3. Blurring**
blurred_img = cv2.blur(img, (5, 5))  # Apply Gaussian blur with kernel size 5x5
# **4. Edge detection**
edges = cv2.Canny(img, 100, 200)  # Apply Canny edge detection
# **5. Thresholding**
ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
# **6. Morphological operations (optional)**
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
# **7. Display or save results**
cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resized_img)

cv2.imshow('Grayscale Image', gray_img)
cv2.imshow('Blurred Image', blurred_img)
cv2.imshow('Edge Detection', edges)
cv2.imshow('Thresholded Image', thresh)
# Optionally display erosion and dilation results
cv2.waitKey(0)
cv2.destroyAllWindows()
# **8. Save results to files (optional)**
cv2.imwrite('resized_image.jpg', resized_img)
cv2.imwrite('grayscale_image.jpg', gray_img)
cv2.imwrite('blur.jpg', blurred_img)
cv2.imwrite('edge_detection.jpg', edges)
cv2.imwrite('threshold.jpg', thresh)
# ... and so on
