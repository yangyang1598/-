import cv2
import numpy as np
from playsound import playsound
from threading import Thread
def play_30():
    playsound('30.mp3')

def play_50():
    playsound('50.mp3')

def play_blue():
    playsound('blue.mp3')

def play_red():
    playsound('red.mp3')

def play_up():
    playsound('speedup.mp3')

def play_down():
    playsound('speeddown.mp3')

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

yolo_net = cv2.dnn.readNet('yolo-obj_last.weights', 'yolo-obj.cfg')
layer_names = yolo_net.getLayerNames()

layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

def Yolo(img, score, nms,pre_label, net, output,speed):
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0),
                                 True, crop=False)
    net.setInput(blob)
    outs = net.forward(output)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 검출 신뢰도
            if confidence > 0.5:
                # Object detected
                # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                dw = int(detection[2] * width)
                dh = int(detection[3] * height)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score, nms)
    for i in range(len(boxes)):

        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            text = f'{label}'
            if label=="Traffic_Light_red":
                if label==pre_label[-1]:
                    pass
                else:
                    R = Thread(target=play_red)  # create thread
                    R.start()
                    # playsound("C:/Users/nrzsd/PycharmProjects/pythonProject2/red.mp3")
                pre_label.append(label)



            if label=="Traffic_Light_green":
                if label==pre_label[-1]:
                    pass
                else:
                    if pre_label[-1] == "sign50" or pre_label[-1]=="sign60":
                        pass
                    else:
                        B = Thread(target=play_blue)  # create thread
                        B.start()

                pre_label.append(label)

            if label=="sign30":
                if label==pre_label[-1]:
                    pass
                else:
                    T = Thread(target=play_30)  # create thread
                    T.start()
                    if speed>30:
                        D=Thread(target=play_down)
                        D.start()
                        speed=30
                    elif speed<30:
                        U = Thread(target=play_up)
                        U.start()
                        speed = 30
                    else:
                        speed=30

                pre_label.append(label)

            if label=="sign50":
                if label==pre_label[-1]:
                    pass
                else:
                    if pre_label[-1] == "Traffic_Light_green" and pre_label[-10:].count("sign50") != 0:
                        pass
                    else:
                        F = Thread(target=play_50)  # create thread
                        F.start()
                        if speed > 50:
                            D = Thread(target=play_down)
                            D.start()
                            speed = 50
                        elif speed < 50:
                            U = Thread(target=play_up)
                            U.start()
                            speed = 50
                        else:
                            speed = 50
                        # playsound("C:/Users/nrzsd/PycharmProjects/pythonProject2/50.mp3")
                pre_label.append(label)


            # 경계상자와 클래스 정보 투영
            cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 2)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_ITALIC, 0.3, (255, 255, 255), 1)
    return img,pre_label,speed
    pass

video=cv2.VideoCapture("Black(2).mp4")
w = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
pre_label=['0']
speed=40
while True:
    _,cap=video.read()
    cap=cv2.pyrDown(cap)
    _,_,speed=Yolo(cap,0.1,0.4,pre_label, yolo_net, output_layers,speed)
    cv2.imshow("hi",cap)
    out.write(cap)
    key=cv2.waitKey(1)
    if key==27:
        video.release()

video.release()
out.release()
cv2.destroyAllWindows()