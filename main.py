import cv2
from fer import FER
import matplotlib.pyplot as plt
import os

def highlightFace(net, frame, conf_threshold = 0.7):
    cv2.imwrite('cam.png', frame)
    test_image_one = plt.imread('D:\\py\\AI_RD\\cam.png')
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
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    captured_emotions = emo_detector.detect_emotions(test_image_one)
    print(captured_emotions)
    plt.imshow(test_image_one)
    dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
    print(dominant_emotion, emotion_score)
    return frameOpencvDnn, faceBoxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
emo_detector = FER(mtcnn=False)

faceNet = cv2.dnn.readNet(faceModel, faceProto)

video = cv2.VideoCapture(0)
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    resultImg, faceBoxes=highlightFace(faceNet, frame)
    cv2.imshow("Face detection", resultImg)