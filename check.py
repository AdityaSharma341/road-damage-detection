from ultralytics.utils.plotting import Annotator
import cv2
import os

img_path = "RoadDataset/train/images/vlcsnap-2025-02-26-20h42m29s611.jpg"
label_path = "RoadDataset/train/labels/vlcsnap-2025-02-26-20h42m29s611.txt"

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(label_path) as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.split())

        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,str(int(cls)),(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,255,0),2)

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()