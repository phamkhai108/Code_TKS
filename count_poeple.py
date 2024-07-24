import cv2
from ultralytics import YOLO

# Khởi tạo
model = YOLO("yolov8n.pt")

video_path = "people.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error accessing the camera"

# Cấu hình
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_position = h // 2  # Đặt đường kẻ ngang ở giữa khung hình
person_count = 0
previous_positions = {}

while True:
    # Đọc và xử lý khung hình
    success, im0 = cap.read()
    if not success:
        print("Camera feed ended.")
        break
    im0 = cv2.flip(im0, 1)  # Lật ảnh theo chiều ngang

    # Theo dõi đối tượng
    results = model.track(im0, persist=True, show=False)

    if results[0].boxes.id is not None:
        for box, id, cls in zip(results[0].boxes.xyxy.cpu().numpy().astype(int),
                                results[0].boxes.id.cpu().numpy().astype(int),
                                results[0].boxes.cls.cpu().numpy().astype(int)):
            if cls == 0:  # Person class
                x1, y1, x2, y2 = box
                center_y = (y1 + y2) // 2

                # Đếm người qua lại
                if id in previous_positions:
                    if previous_positions[id] >= line_position > center_y:
                        person_count += 1
                        cv2.putText(im0, "UP", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif previous_positions[id] < line_position <= center_y:
                        person_count = max(0, person_count - 1)
                        cv2.putText(im0, "DOWN", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                previous_positions[id] = center_y

                # Vẽ bounding box
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0, f'Person #{id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vẽ và hiển thị
    cv2.line(im0, (0, line_position), (im0.shape[1], line_position), (0, 0, 255), 2)
    cv2.putText(im0, f'Current people count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Object Counting", im0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()
