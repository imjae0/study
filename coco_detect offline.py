import torch
import cv2
import numpy as np
import tensorflow as tf
from pybboxes import convert_bbox



# model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/admin/Desktop/,/test/debtolee/number_YOLOv5_best.pt', force_reload=True)
model = torch.hub.load('C:/Users/admin/Desktop/,/test/yolov5', 'custom', path='C:/Users/admin/Desktop/,/test/debtolee/number_YOLOv5_best.pt', source="local")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'div', 'eqv', 'minus', 'mult', 'plus']
cap = cv2.VideoCapture(0)   

half = device.type != 'cpu'
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,400)
CONF_THRES = 0.15
IOU_THRES = 0.25

while(True):
    ret, img = cap.read()
#    cam = cv2.imread("yolov5-master/coco_dataset/test/images/0ZGdgPTM_png_jpg.rf.6c4c65eeddad1d6155d78eda475a4e14.jpg")
    if(ret) :
        img2 = img.copy()
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # RGB 채널
        print(img2)
        img2 = img2.astype(np.uint8).copy() # np.float32 -> np.uint8
        print(img2)
        # break
        pred = model(img2)
        results = pred.xyxy

        for res in results[0]:

            if float(res[4].item()) < 0.6:
                continue
            bb = res[0:4].tolist()

            label = str(class_names[int(res[-1].item())])
            label = str(class_names[int(res[-1].item())])
            x,y,w,h = list(map(int, bb))
            print(label,x,y,w,h)
            img2 = cv2.rectangle(img2, (x, y), (w,  h), (36,255,12), 1)
            cv2.putText(img2, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#         except:
#             pass
        cv2.imshow('camera', img2)

        if cv2.waitKey(1) & 0xFF == 27: # esc 키를 누르면 닫음
            break

cap.release()
cv2.destroyAllWindows()