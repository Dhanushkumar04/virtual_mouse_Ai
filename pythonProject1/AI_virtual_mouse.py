import cv2
import numpy as np
import hand_tracking as htm
import time
import pyautogui

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 5  # Increased smoothening factor
click_distance_threshold = 30  # Distance to trigger click
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
click_delay = 0.3  # Time delay between clicks to debounce
last_click_time = 0  # Track last click time

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if the video capture is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.5)  # Adjusted detection confidence
wScr, hScr = pyautogui.size()  # Get the screen size

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers, only if landmarks are detected
    if len(lmList) > 12:
        x1, y1 = lmList[8][1], lmList[8][2]  # Index finger
        x2, y2 = lmList[12][1], lmList[12][2]  # Middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY  # Update previous location

        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # 10. Click mouse if distance short
            if length < click_distance_threshold:
                current_time = time.time()
                # Check if enough time has passed since the last click
                if current_time - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = current_time  # Update last click time
                    cv2.circle(img, (lineInfo[2], lineInfo[3]), 15, (0, 255, 0), cv2.FILLED)

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
