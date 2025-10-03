from ultralytics import YOLO
import cv2
import cvzone
import math
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
model=YOLO('../yolo-weighs/yolov8l.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","waterbottle","pen","apple","orange","laserlight","comb","note","lion","saipallavi","fan","light","door","Mirror","caromboard","window","cupboard"
              ]
while(True):
    sucess,image=cap.read()
    results=model(image,stream=True)
    # 'results' is a generator of Results objects, each containing detections for one image/frame
    # Results objects hold bounding boxes, class labels, scores, and masks as PyTorch tensors
    #******************Bounding box******************************
    for r in results:
        boxes = r.boxes  # Extract bounding boxes detected in this frame
        for box in boxes:  # Loop over each detected bounding box
            # xyxy tensor stores absolute pixel coordinates of the box corners:
            # x1, y1 = top-left corner; x2, y2 = bottom-right corner
            x1, y1, x2, y2 = box.xyxy[0]
            # Convert tensor float values to integer pixel coordinates for further use
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # Further processing like drawing bounding boxes or saving can be done here
            #--------------------using cvzone--------------------
            #w,h=x2-x1,y2-y1
            #cvzone.cornerRect(image,(x1,y1,w,h))
            #----------------------------------------------------

            #-------------------using cv2------------------------
            print(x1,y1,x2,y2)
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,0),2)
            #----------------------------------------------------

            #*************************confidence interval*************************

            conf=math.ceil(box.conf[0]*100)/100
            print(conf)
            #*************************class name***************************************
            cls=int(box.cls[0])
            cvzone.putTextRect(image,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1, thickness=1)

    cv2.imshow("image", image)
    cv2.waitKey(1)
