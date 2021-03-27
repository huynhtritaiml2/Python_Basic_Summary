import cv2
import numpy as np

cap = cv2.VideoCapture(0)
'''
0 : camera 0
1 : camera 1
'''
while True:
    ret, frame = cap.read()
    # ret: True: can use camera, False: cannot
    # frame: image from camera 

    height = int(cap.get(4)) # 4: is height
    width = int(cap.get(3)) # 3: is width
    # NOTE: cap.get() return float
    print(type(frame)) # <class 'numpy.ndarray'>
    print(frame.shape) # (480, 640, 3)
    print(f"({height}, {width})") # (480, 640)
    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    
    image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, :width//2] = smaller_frame
    image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, width//2:] = smaller_frame

    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord('q'):
        # wait 1ms and during that ime if q is pressed, ord() return ASCII integer 
        break

cap.release() # Other software can use this devices
cv2.destroyAllWindows()
    
