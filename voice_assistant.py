import sounddevice as sd
import vosk
import queue
import json
import socket
import speech_recognition as sr
import pyttsx3
import threading
import time
import logging

from config import MODEL_PATH, SAMPLE_RATE, DEVICE

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoiceAssistant:
    def __init__(self):
        self.model = vosk.Model(MODEL_PATH)
        self.queue = queue.Queue()
        self.stop_listening = None
        self.is_online = False
        self.online_thread = None
        self.offline_thread = None

    def check_internet(self, host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except Exception as ex:
            logging.error(f"Проблема с интернетом: {ex}")
            return False

    def callback_vosk(self, indata, frames, time, status):
        if status:
            logging.error(status)
        self.queue.put(bytes(indata))

    def speak(self, text):
        def run_speak():
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('voice', 'ru')
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=run_speak).start()

    def callback_google(self, recognizer, audio):
        try:
            voice = recognizer.recognize_google(audio, language="ru-RU").lower()
            logging.info(f"Распознано: {voice}")
            self.speak(voice)
        except sr.UnknownValueError:
            logging.warning("Голос не распознан!")
        except sr.RequestError:
            logging.error("Неизвестная ошибка, проверьте интернет!")

    def online_listener(self):
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
        
        with sr.Microphone(device_index=None) as source:
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source)
            recognizer.pause_threshold = 0.5  # Уменьшено время ожидания после завершения речи
            recognizer.operation_timeout = 3  # Установите максимальное время ожидания

        self.stop_listening = recognizer.listen_in_background(source, self.callback_google)
        try:
            while self.is_online:
                time.sleep(0.1)
                self.is_online = self.check_internet()
        finally:
            if self.stop_listening:
                self.stop_listening(wait_for_stop=False)


    def offline_listener(self):
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE, 
            blocksize=8000, 
            device=DEVICE, 
            dtype='int16',
            channels=1, 
            callback=self.callback_vosk
        ):
            recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
            logging.info("Начинаем запись, говорите...")
            while not self.is_online:
                data = self.queue.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    text = json.loads(result).get('text', '')
                    self.speak(text)
                    logging.info(f"Распознанный текст: {text}")
                else:
                    partial_result = recognizer.PartialResult()
                    logging.debug(partial_result)

    def start(self):
        while True:
            self.is_online = self.check_internet()
            if self.is_online:
                if self.online_thread and self.online_thread.is_alive():
                    continue
                self.online_thread = threading.Thread(target=self.online_listener)
                self.online_thread.start()
            else:
                if self.offline_thread and self.offline_thread.is_alive():
                    continue
                self.offline_thread = threading.Thread(target=self.offline_listener)
                self.offline_thread.start()