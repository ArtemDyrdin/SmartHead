import threading

from voice_assistant import VoiceAssistant
from yolo_detector import YOLODetector


def main():
    # Создание и запуск голосового ассистента
    assistant = VoiceAssistant()

    # Создание детектора объектов
    detector = YOLODetector("yolo_data/yolov4-tiny.cfg", "yolo_data/yolov4-tiny.weights", "yolo_data/coco.names")

    # Запуск голосового ассистента в отдельном потоке
    assistant_thread = threading.Thread(target=assistant.start)
    assistant_thread.start()

    # Запуск обнаружения объектов в отдельном потоке
    detector_thread = threading.Thread(target=detector.process_video(0))
    detector_thread.start()

if __name__ == "__main__":
    main()
