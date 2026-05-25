# Basic asf but works perfectly. Takes image (or a frame from a video feed) and masks out the ring by filtering out its color. Then finds the average of the position of the masked out pixels
# Running this in google colab is easiest. Open a new colab notebook, paste this all into one cell, and import the image (in this directory) into the file menu on the left. Then run (first run takes a while because it has to connect to googel servers)
# The image rn is just a pic from the tech spec, but once the sim releases we cna troubleshoot with real data


import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread("aigp_image.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Color threshold for gate
lower = np.array([110, 100, 50])
upper = np.array([130, 255, 255])

mask = cv2.inRange(hsv, lower, upper)

# Remove noise (no noise in this image)
kernel = np.ones((5,5), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 500:
        continue

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:

        x, y, w, h = cv2.boundingRect(approx)

        center_x = x + w // 2
        center_y = y + h // 2

        print("Center:", center_x, center_y)

        cv2.drawContours(img, [approx], -1, (0,255,0), 3)
        cv2.circle(img, (center_x, center_y), 8, (0,0,255), -1)

cv2_imshow(mask)
cv2_imshow(img)
