import numpy as np
import cv2
from PIL import ImageGrab
import time
import matplotlib.pyplot as plt
from lane_findings import process_image

def screen_grab():
    last_time = time.time()
    #image = []
    while True:
        printScreen = np.array(ImageGrab.grab(bbox = (0,40,670,503)))
        print('Loop took {} seconds .'.format(time.time() - last_time))
        last_time  = time.time()
        screen = process_image(printScreen)
        cv2.imshow('window', screen)
        #image.append(printScreen)
        #cv2.imshow('window', cv2.cvtColor(printScreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    #cv2.imwrite('image.png',np.array(image[7], dtype='float32'))

screen_grab()
#plt.imshow('image.png')
