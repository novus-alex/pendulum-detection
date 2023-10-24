import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.signal import find_peaks

def detect(filename, wl):
    cap = cv2.VideoCapture(filename)
    save = []
    T = []
    i = 0
    
    while(1):
        _, frame = cap.read()
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        except:
            break

        lower1 = np.array([80, 100, 140])
        upper1 = np.array([90, 255, 255])

        lower2 = np.array([100, 100, 140])
        upper2 = np.array([110, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        result1 = cv2.bitwise_and(frame, frame, mask = mask1)

        mask2 = cv2.inRange(hsv, lower2, upper2)
        result2 = cv2.bitwise_and(frame, frame, mask = mask2)

        if len(save) > 0:
            #diff = cv2.absdiff(result, save)
            thresh1 = cv2.threshold(result1, 230, 240, cv2.THRESH_BINARY)[1]
            thresh1 = cv2.dilate(thresh1, None, iterations=2)

            thresh2 = cv2.threshold(result2, 240, 240, cv2.THRESH_BINARY)[1]
            thresh2 = cv2.dilate(thresh2, None, iterations=2)
            
            x, y = np.where(np.all(thresh1!=[240, 240, 240],axis=2))
            xp, yp = np.where(np.all(thresh2!=[240, 240, 240],axis=2))
            dX, dY = np.sum(x)/len(x) - np.sum(xp)/len(xp), np.sum(y)/len(y) - np.sum(yp)/len(yp)

            #print(dX, dY)

            if dY != 0:
                T.append(atan(dY/dX)-0.3)
            else:
                T.append(pi/2)

            
            cv2.imshow("r1", thresh1)
            cv2.imshow("r2", thresh2)
            cv2.imshow("vid", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        """
        if i == 100:
            cv2.imwrite("vid.png", frame)
            cv2.imwrite("t1.png", thresh1)
            cv2.imwrite("t2.png", thresh2)
        """
        

        save = result1
        i += 1
        
    cv2.destroyAllWindows()
    cap.release()

    return T

#plt.plot([i/30 for i in range(len(T))], 0.001 + max(T[30:])*np.exp(-0.3*np.array([i/30 for i in range(len(T))])))
#plt.plot([i/30 for i in range(len(T))], T)
T = np.array(detect("IMG_2211.MOV", 500))


FFT = np.fft.fft(T[66:], n=1000)
freq = np.fft.fftfreq(T.size, d=0.1)

plt.style.use(["science"])

fig, ax = plt.subplots(2, 2)
ax[0][0].plot([0.1*i for i in range(len(T[66:]))], T[66:], "k", lw=0.8)
ax[0][0].set_ylabel(r"$\theta$")
ax[1][0].plot(freq[:100], np.abs(FFT)[:len(freq)][:100], "k", lw=0.8)

ax[0][1].plot(T[66:-1], [(T[i+1] - T[i])/0.1 for i in range(66, len(T)-1)])
plt.show()
