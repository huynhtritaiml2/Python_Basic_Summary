import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    height = int(cap.get(4))
    width = int(cap.get(3))

    img = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 10)
    # Staring point, Ending point, color, thickness 
    #img = cv2.line(img, (width, 0), (0, height), (0, 255, 0), 5) # Starting point and Ending point we can swap it
    img = cv2.line(img, (0, height), (width, 0), (0, 255, 0), 5)


    img = cv2.rectangle(img, (100, 100), (200, 200), (128, 128, 128), 5)
    # Staring point, Ending point, color, thickness 

    img = cv2.circle(img, (300, 300), 60, (0, 0, 255), -1)
    # the center point, radius, color, thickness (-1 to fill)


    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, "Tim is Great!", (10, height - 10), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
    # Text, position, font, font size, color, thickness, aligned-type
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
