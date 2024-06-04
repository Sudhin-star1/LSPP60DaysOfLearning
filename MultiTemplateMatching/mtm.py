import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

img = cv2.imread('MultiTemplateMatching/gfg.png')
temp = cv2.imread('MultiTemplateMatching/template.jpeg')

W, H = temp.shape[:2]

thresh = 0.4

img_gray = cv2.cvtColor(img,
                        cv2.COLOR_BGR2GRAY)
temp_gray = cv2.cvtColor(temp,
                         cv2.COLOR_BGR2GRAY)

match = cv2.matchTemplate(
    image=img_gray, templ=temp_gray,
    method=cv2.TM_CCOEFF_NORMED)


(y_points, x_points) = np.where(match >= thresh)

boxes = list()

for (x, y) in zip(x_points, y_points):
    boxes.append((x, y, x + W, y + H))

boxes = non_max_suppression(np.array(boxes))

for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(img, (x1, y1), (x2, y2),
                  (255, 0, 0), 3)

cv2.imshow("Template", temp)
cv2.imshow("Image", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
