import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []
        self.tipIds = [4, 8, 12, 16, 20]  # Tips of the fingers

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        self.results = self.hands.process(imgRGB)  # Process the image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])  # Append the coordinates to lmList
            if draw:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) > 0:  # Check if lmList has any elements
            # Thumb
            fingers.append(1 if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1] else 0)
            # Other fingers
            for id in range(1, 5):
                if len(self.lmList) > self.tipIds[id]:  # Check if index is valid
                    fingers.append(1 if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 1][2] else 0)
                else:
                    fingers.append(0)  # Append 0 if lmList is not long enough
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        if len(self.lmList) > p1 and len(self.lmList) > p2:  # Check if both points are in lmList
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)  # Calculate distance
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Draw line
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # Draw circle
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)  # Draw circle
            return length, img, [x1, y1, x2, y2]  # Return length and img
        return 0, img, [0, 0, 0, 0]  # Return 0 if points not found

    def start_video_capture(self):
        cap = cv2.VideoCapture(0)  # Start capturing from the camera
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            success, img = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break

            img = self.findHands(img)  # Find hands
            lmList = self.findPosition(img)  # Find positions

            if lmList:  # If landmarks are found, print them
                print(lmList)

            # Optional: Add any processing or drawing here

            cv2.imshow("Hand Tracking", img)  # Show the image
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
                break

        cap.release()  # Release the capture
        cv2.destroyAllWindows()  # Close all OpenCV windows


# If you want to run the video capture when this script is executed
if __name__ == "__main__":
    detector = HandDetector(maxHands=1, detectionCon=0.5, trackCon=0.5)
    detector.start_video_capture()
