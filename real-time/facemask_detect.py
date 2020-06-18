# Face Mask Detection Real-time
import cv2
from imutils.video import VideoStream
import time 
import imutils
import numpy as np
from gtts import gTTS
from playsound import playsound
from threading import Thread
import time
import face_recognition

yolo_model = './project-pack/yolov3-tiny-mask_best-11062020.weights'
yolo_cfg = './project-pack/yolov3-tiny-mask.cfg'
yolo_names = './project-pack/train_data/obj_mask.names'
yolo_confidence = 0.6
yolo_threshold = 0.3
known_encoding = []
known_person = []
def loadFaces():
    global known_encoding,known_person
    img = face_recognition.load_image_file("monicaleung.jpg")
    image_encoding = face_recognition.face_encodings(img)[0]
    known_encoding = [
     image_encoding,
    ]
    known_person = [
        "Monica"
    ]


isPlayed = False
def speak(text):
    global isPlayed
    tts = gTTS(text=text, lang='en')
    tts.save("tmp.mp3")
    playsound("tmp.mp3")
    time.sleep(1)
    isPlayed = False

W = None
H = None


def load_label():
    LABELS = open(yolo_names).read().strip().split("\n")
    return LABELS

def load_yolo():
    net = cv2.dnn.readNetFromDarknet(yolo_cfg,yolo_model)
    return net
    

## Main
net = load_yolo()
LABELS = load_label()
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
    
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = VideoStream(src=0).start()
time.sleep(2.0)
totalFrames = 0
loadFaces()
while True:
    frame = vs.read()
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:

            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > yolo_confidence:
            # scale the bounding box coordinates back relative to
            # the size of the image, keeping in mind that YOLO
            # actually returns the center (x, y)-coordinates of
            # the bounding box followed by the boxes' width and
            # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, yolo_confidence, yolo_threshold)
    # print(len(idxs))
    if len(idxs) > 0:
        for i in idxs.flatten():
                        # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            
                        # draw a bounding box rectangle and label on the frame
                        # color = [int(c) for c in COLORS[classIDs[i]]]
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            if classIDs[i] == 0:
                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 1.5, color 
                , 4)
            else:
                cropped = frame[y-20:y+h+20,x-20:x+h+20]
                cropped_rgb = cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB)
                cv2.imshow("cropped",cropped)
            
                # img_unknown = face_recognition.load_image_file(cropped_rgb)
                unknown_encoding = face_recognition.face_encodings(cropped_rgb)
                name = "Unknown Person"
                if len(unknown_encoding) > 0:
                    unknown_encoding = unknown_encoding[0]
                    face_distances = face_recognition.face_distance(known_encoding,unknown_encoding)
                    print(face_distances)
                    for i, face_distance in enumerate(face_distances):
                        if face_distance < 0.65:
                            name = known_person[i]
                
                
            #    print(isPlayed)
                if isPlayed is False :
                    isPlayed = True
                    Thread(target=speak,args=[name +"Please wear mask"]).start()
                   
                text = "{}: {:.4f}".format("Wear Mask", confidences[i])
                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255) 
                , 4) 

    
    totalFrames += 1
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# HouseKeeping
vs.stop()
cv2.destroyAllWindows()
