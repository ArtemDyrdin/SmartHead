import cv2
import numpy as np
import time

class YOLODetector:
    def __init__(self, model_cfg, model_weights, classes_file):
        self.net = cv2.dnn.readNet(model_weights, model_cfg)
        with open(classes_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    # Функция обработки видеопотока и обнаружения объектов
    def process_video(self, stream_source=0, confidence_threshold=0.5, nms_threshold=0.4):
        cap = cv2.VideoCapture(stream_source)
        prev_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            height, width = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids, confidences, boxes = [], [], []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        center_x, center_y, w, h = (detection[0] * width, detection[1] * height,
                                                    detection[2] * width, detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, int(w), int(h)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

            if indexes is not None and len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = f"{self.classes[class_ids[i]]} {confidences[i]:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            fps = 1 / (time.time() - current_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Image", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Нажмите ESC для выхода
                break

        cap.release()
        cv2.destroyAllWindows()

# # Функция загрузки YOLO модели и классов
# def load_yolo(model_cfg, model_weights, classes_file):
#     net = cv2.dnn.readNet(model_weights, model_cfg)
#     with open(classes_file, "r") as f:
#         classes = [line.strip() for line in f.readlines()]
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#     return net, classes, output_layers

