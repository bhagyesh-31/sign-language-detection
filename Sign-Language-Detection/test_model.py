import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Camera setup
cap = cv2.VideoCapture(0)

# Hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Configuration
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop hand image
        imgCrop = img[
            y - offset : y + h + offset,
            x - offset : x + w + offset
        ]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)

            imgWhite[:, wGap : wCal + wGap] = imgResize

            prediction, index = classifier.getPrediction(
                imgWhite, draw=False
            )
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap : hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(
                imgWhite, draw=False
            )

        # Display label background
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset - 50),
            (x - offset + 90, y - offset),
            (255, 0, 255),
            cv2.FILLED
        )

        # Display label text
        cv2.putText(
            imgOutput,
            labels[index],
            (x, y - 26),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 255, 255),
            2
        )

        # Draw bounding box
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x + w + offset, y + h + offset),
            (255, 0, 255),
            4
        )

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
