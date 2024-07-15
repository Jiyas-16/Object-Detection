import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)

class_names = []
with open('coco.names', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

object_configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
object_weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(object_weightsPath, object_configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

face_cascade_path = 'haarcascade_frontalface_default.xml'
face_detection_model = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

if face_detection_model.empty():
    raise Exception(f'Error loading the face detection model: {face_cascade_path}')

cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

while True:

    success, img = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 27:
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    class_ids, confidences, boxes = net.detect(img, confThreshold=0.5)

    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            color = (0, 255, 0)
            cv2.rectangle(img, box, color, thickness=3)
            class_name = class_names[class_id - 1].upper()
            label = f'{class_name} ({confidence:.2f})'
            label_bg_color = (0, 0, 0)
            label_text_color = (255, 255, 255)
            cv2.rectangle(img, (box[0], box[1] - 30), (box[0] + len(label) * 17, box[1]), label_bg_color, -1)
            cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 2)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection_model.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Object Detection", img)

cap.release()
cv2.destroyAllWindows()
