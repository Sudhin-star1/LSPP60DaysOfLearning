import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
confidence_threshold = 0.5

video_path = 'YOLOv8/vid.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = 'output.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc,fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detected_objects = results[0].boxes.data

    detected_objects = detected_objects[detected_objects[:, 4] > confidence_threshold]

    for obj in detected_objects:
        class_id = int(obj[5])
        label = model.names[class_id]
        confidence = obj[4].item()
        bbox = obj[:4].int().tolist()
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[2], bbox[3])

        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

        text = f'{label} {confidence:.2f}'
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)
    # Display the frame with bounding boxes and labels
    cv2.imshow('YOLOv8 Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
# Close OpenCV windows
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("HI")
