import cv2
import mediapipe as mp 
import time 


class HandDetector(): 
    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon= 0.5):
        self.mode = mode
        self.maxhands = maxhands 
        self.detectioncon = detectioncon
        self.trackcon = trackcon
        

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxhands, min_detection_confidence=self.detectioncon, min_tracking_confidence=self.trackcon)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw= True): 
        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                lmlist.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return lmlist 

def main(): 
    previous_time = 0 
    current_time = 0 
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True: 
        success, img = cap.read()
        img = detector.findhands(img, draw=False)
        lmlist = detector.findPosition(img, draw=False)
        if len(lmlist) !=0:
            print(lmlist[4])

        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 1)


        cv2.imshow ("image", img)
        cv2.waitKey(1) 

if __name__ == "__main__": 
    main()