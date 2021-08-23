import cv2 as cv
import mediapipe as mp

cap =cv.VideoCapture(0,cv.CAP_DSHOW)

mpHands=mp.solutions.hands
hands=mpHands.Hands()

mpDraw=mp.solutions.drawing_utils



while True:
    ret,frame=cap.read()

    frame=cv.flip(frame,2)

    framegray=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    result=hands.process(framegray)

    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for mhandl in result.multi_hand_landmarks:

            for id,lm in enumerate(mhandl.landmark):

                h,w,c=frame.shape

                cx,cy=int(lm.x*w),int(lm.y*h)

                if id==4:

                    cv.circle(frame,(cx,cy),15,(0,255,0),-1)

            mpDraw.draw_landmarks(frame,mhandl,mpHands.HAND_CONNECTIONS)








    cv.imshow("image",frame)

    if cv.waitKey(5) & 0xFF==ord("q"):
        break





cv.destroyAllWindows()