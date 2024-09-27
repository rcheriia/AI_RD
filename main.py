import cv2
from fer import FER
import matplotlib.pyplot as plt
import os
import time

colors = {'angry': (0, 0, 255), 'disgust': (0, 255, 68), 'fear': (245, 5, 189), 'happy': (3, 234, 255), 'sad': (255, 0, 0), 'surprise': (0, 111, 255), 'neutral': (0, 0, 0)}
def highlightFace(net, frame, Emo, conf_threshold = 0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence=detections[0, 0, i, 2]
        if confidence>conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 0, 255), int(round(frameHeight / 150)), 8)
            cv2.rectangle(frameOpencvDnn, (x1, y2 + 10), (x2, y2 + 40), (255, 255, 255), int(round(frameHeight / 150)), 8)
            cv2.putText(frameOpencvDnn, Emo, (x1 + 50, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[Emo.split()[0]], 2)
    return frameOpencvDnn, faceBoxes

def emg(frame):
    cv2.imwrite('cam.jpg', frame)
    test_image_one = plt.imread('D:\\py\\AI_RD\\cam.jpg')
    plt.imshow(test_image_one)
    dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
    emotion_score = float(emotion_score)
    return f"{dominant_emotion} {int(emotion_score * 100.0)}%"

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
emo_detector = FER(mtcnn=True)

faceNet = cv2.dnn.readNet(faceModel, faceProto)

video = cv2.VideoCapture(0)
current_time = time.time()
formatted_time = time.ctime(current_time)
p_t = int(formatted_time.split()[3].split(':')[2]) + 1
Emo = 'happy 100%'
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    current_time = time.time()
    formatted_time = time.ctime(current_time)
    tim = formatted_time.split()[3]
    if int(tim.split(':')[2]) - 1 == p_t:
        Emo = emg(frame)
        p_t = int(tim.split(':')[2])
    if not hasFrame:
        cv2.waitKey()
        break
    resultImg, faceBoxes=highlightFace(faceNet, frame, Emo)
    cv2.imshow("Face detection", resultImg)

