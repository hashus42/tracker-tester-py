import cv2 as cv
import numpy as np

tracker = cv.TrackerDaSiamRPN.create()

cap = cv.VideoCapture("data/amman-stunt.mp4")
ret, frame = cap.read()
if not ret:
    exit(-1)

# initialize tracker
bbox = cv.selectROI('Tracker', frame)
tracker.init(frame, bbox)

while True:
    start = cv.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break

    # track
    output = tracker.update(frame)

    # draw bounding box
    p1 = (int(output[1][0]), int(output[1][1]))
    p2 = (int(output[1][0] + output[1][2]), int(output[1][1] + output[1][3]))
    cv.rectangle(frame, p1, p2, (0, 255, 0), 2)

    # FPS
    fps = cv.getTickFrequency() / (cv.getTickCount() - start)
    cv.putText(frame, 'FPS : ' + str(int(fps)), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)

    # display
    cv.imshow('Tracker', frame)

    # Exit on ESC
    k = cv.waitKey(1) & 0xff
    if k == 27: break

# Close the window / Release handle if required
cv.destroyAllWindows()
