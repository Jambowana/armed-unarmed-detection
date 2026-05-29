import cv2
from ultralytics import YOLO

person_model = YOLO("yolov8n.pt")
gun_model = YOLO(r"C:\Users\ASUS\runs\detect\train5\weights\best.pt")

def is_overlapping(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x2 > x1 and y2 > y1

def face_to_body(x, y, w, h):
    return [x - w, y, x + w * 2, y + h * 4]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    gun_results = gun_model(frame, conf=0.5, verbose=False)
    gun_boxes = []
    if gun_results[0].boxes is not None:
        for box in gun_results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            gun_boxes.append([x1, y1, x2, y2, conf])

    for (x, y, w, h) in faces:
        body_box = face_to_body(x, y, w, h)
        armed = False

        for gun_box in gun_boxes:
            if is_overlapping(body_box, gun_box[:4]):
                armed = True
                break

        color = (0, 0, 255) if armed else (0, 255, 0)
        label = "ARMED" if armed else "UNARMED"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        bx1, by1, bx2, by2 = body_box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)

    for gun_box in gun_boxes:
        x1, y1, x2, y2, conf = gun_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"Gun {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow("Armed/Unarmed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()