import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model_1.p', 'rb'))
model = model_dict['model']

labels_dict = ['T', 'ThankYou', '3', '2', 'No', 'Friend', 'X', 'Hello', '1', 'Try', 'A', 'B', 'D', 'L', '9', '4', 'C']



url = "http://192.168.213.214:8080/video"
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands#type:ignore
mp_drawing = mp.solutions.drawing_utils#type:ignore
mp_drawing_styles = mp.solutions.drawing_styles#type:ignore

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3,max_num_hands=4)


frame_height = 600
frame_width = 400
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        break


    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])


        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, prediction[0], (x2+10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 6, (90, 21, 255), 14,
                    cv2.LINE_AA)
        
    text = "By NaniK & company.."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3

    height,width,_=frame.shape
    (text_width,text_height),baseline = cv2.getTextSize(text,font,font_scale,thickness)

    x = width-text_width-40
    y = height-10

    cv2.putText(frame,text,(x,y),font,font_scale,(90,21,255),thickness,cv2.LINE_AA)

    cv2.imshow('frame', cv2.resize(frame,(frame_height,frame_width)))
    cv2.waitKey(1)


    if cv2.waitKey(1)==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()