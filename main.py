#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# НАСТРОЙКИ KIVY - ДОЛЖНЫ БЫТЬ ПЕРЕД ИМПОРТАМИ!
import os
import sys

# Решение проблемы OpenGL
os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'
os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_VIDEO'] = 'ffpyplayer'
os.environ['KIVY_AUDIO'] = 'ffpyplayer'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Подавляем предупреждения OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

from kivy.config import Config

Config.set('graphics', 'multisamples', '0')
Config.set('graphics', 'backend', 'sdl2')
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '768')
Config.set('graphics', 'resizable', True)
Config.set('input', 'mouse', 'mouse,disable_multitouch')
Config.set('kivy', 'exit_on_escape', '0')
Config.set('graphics', 'maxfps', '30')

# Теперь можно импортировать Kivy модули
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.utils import platform
from kivy.metrics import dp
from kivy.loader import Loader
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Rectangle
from kivy.uix.progressbar import ProgressBar
from kivy.uix.relativelayout import RelativeLayout

# Импорты для компьютерного зрения и ИИ
import cv2
import numpy as np
import datetime
import json
import warnings
from pathlib import Path
import threading
import time

# Определяем, запущены ли мы в эмуляторе Android на Windows
IN_ANDROID_EMULATOR = False
if os.path.exists('/system/build.prop') or 'ANDROID_ROOT' in os.environ:
    IN_ANDROID_EMULATOR = True
    print("⚠ Обнаружен Android-эмулятор")
    print("⚠ Камера может быть недоступна, используем демо-режим")

# TensorFlow импорты с обработкой ошибок
try:
    import tensorflow as tf
    from tensorflow import keras
    import joblib

    TF_AVAILABLE = True
    print("✓ TensorFlow загружен")
    print(f"  Версия TensorFlow: {tf.__version__}")
except ImportError as e:
    TF_AVAILABLE = False
    print(f"✗ TensorFlow не загружен: {e}")
    print("  Работа в демо-режиме без ИИ")

# Отключаем предупреждения TensorFlow
warnings.filterwarnings('ignore')
if TF_AVAILABLE:
    tf.get_logger().setLevel('ERROR')


class UserProfile:
    """Класс для хранения и управления профилем пользователя"""

    def __init__(self):
        self.profile_file = "user_profile.json"
        self.last_max_update_file = "last_max_update.json"
        self.start_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.total_days = 1
        self.total_workouts = 0
        self.total_reps = {
            "отжимания": 0,
            "подтягивания": 0,
            "пресс": 0,
            "приседания": 0
        }
        self.max_reps = {
            "отжимания": 0,
            "подтягивания": 0,
            "пресс": 0,
            "приседания": 0
        }
        self.original_max = {
            "отжимания": 0,
            "подтягивания": 0,
            "пресс": 0,
            "приседания": 0
        }
        self.current_targets = {
            "отжимания": 0,
            "подтягивания": 0,
            "пресс": 0,
            "приседания": 0
        }
        self.workout_history = []
        self.last_workout_date = None
        self.streak_days = 0
        self.last_max_update = None

        self.load_profile()
        self.load_last_max_update()

    def load_profile(self):
        """Загрузка профиля из файла"""
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.start_date = data.get('start_date', self.start_date)
                self.total_days = data.get('total_days', 1)
                self.total_workouts = data.get('total_workouts', 0)
                self.total_reps = data.get('total_reps', self.total_reps)
                self.max_reps = data.get('max_reps', self.max_reps)
                self.original_max = data.get('original_max', self.max_reps.copy())
                self.current_targets = data.get('current_targets', self.max_reps.copy())
                self.workout_history = data.get('workout_history', [])
                self.last_workout_date = data.get('last_workout_date')
                self.streak_days = data.get('streak_days', 0)

                print("✓ Профиль пользователя загружен")
            except Exception as e:
                print(f"Ошибка загрузки профиля: {e}")

    def save_profile(self):
        """Сохранение профиля в файл"""
        try:
            data = {
                'start_date': self.start_date,
                'total_days': self.total_days,
                'total_workouts': self.total_workouts,
                'total_reps': self.total_reps,
                'max_reps': self.max_reps,
                'original_max': self.original_max,
                'current_targets': self.current_targets,
                'workout_history': self.workout_history[-100:],  # Храним последние 100 тренировок
                'last_workout_date': self.last_workout_date,
                'streak_days': self.streak_days
            }

            with open(self.profile_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print("✓ Профиль пользователя сохранен")
        except Exception as e:
            print(f"Ошибка сохранения профиля: {e}")

    def load_last_max_update(self):
        """Загрузка даты последнего обновления максимумов"""
        if os.path.exists(self.last_max_update_file):
            try:
                with open(self.last_max_update_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.last_max_update = data.get('last_update')
                print("✓ Дата последнего обновления максимумов загружена")
            except Exception as e:
                print(f"Ошибка загрузки даты обновления: {e}")

    def save_last_max_update(self):
        """Сохранение даты последнего обновления максимумов"""
        try:
            data = {
                'last_update': datetime.datetime.now().strftime("%Y-%m-%d")
            }
            with open(self.last_max_update_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.last_max_update = data['last_update']
            print("✓ Дата обновления максимумов сохранена")
        except Exception as e:
            print(f"Ошибка сохранения даты обновления: {e}")

    def needs_max_update(self):
        """Проверяет, нужно ли обновить максимумы (прошла неделя)"""
        if self.last_max_update is None:
            return True

        try:
            last = datetime.datetime.strptime(self.last_max_update, "%Y-%m-%d")
            today = datetime.datetime.now()
            days_passed = (today - last).days
            return days_passed >= 7
        except:
            return True

    def update_max_values(self, new_max_values):
        """Обновляет максимальные значения и пересчитывает цели"""
        self.original_max = new_max_values.copy()
        self.max_reps = new_max_values.copy()

        # Устанавливаем начальные цели (равны максимуму)
        for exercise in new_max_values:
            self.current_targets[exercise] = new_max_values[exercise]

        self.save_last_max_update()
        self.save_profile()
        print("✓ Максимальные значения обновлены")

    def increase_target(self, exercise):
        """Увеличивает цель для упражнения на 2.5 от максимального"""
        if exercise in self.current_targets and exercise in self.original_max:
            # Добавляем 2.5 от исходного максимума
            increment = int(self.original_max[exercise] * 2.5)
            self.current_targets[exercise] += increment
            print(f"✓ Цель для {exercise} увеличена на {increment} (теперь {self.current_targets[exercise]})")
            self.save_profile()
            return True
        return False

    def add_workout(self, exercise, reps):
        """Добавление тренировки"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        # Обновляем общую статистику
        self.total_workouts += 1
        self.total_reps[exercise] += reps

        # Обновляем максимальные повторения (если новое значение больше)
        if reps > self.max_reps[exercise]:
            self.max_reps[exercise] = reps

        # Обновляем историю
        self.workout_history.append({
            'date': today,
            'exercise': exercise,
            'reps': reps
        })

        # Обновляем streak
        if self.last_workout_date == today:
            pass  # Уже тренировались сегодня
        elif self.last_workout_date == self.get_yesterday():
            self.streak_days += 1
        else:
            self.streak_days = 1

        self.last_workout_date = today

        # Обновляем общее количество дней
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        today_dt = datetime.datetime.now()
        self.total_days = (today_dt - start).days + 1

        self.save_profile()

    def get_yesterday(self):
        """Получение вчерашней даты"""
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")

    def get_week_stats(self):
        """Получение статистики за последние 7 дней"""
        week_stats = {ex: 0 for ex in self.total_reps.keys()}

        today = datetime.datetime.now()
        for i in range(7):
            day = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            for workout in self.workout_history:
                if workout['date'] == day:
                    week_stats[workout['exercise']] += workout['reps']

        return week_stats


class CameraStream:
    """Класс для асинхронной работы с камерой (с поддержкой эмулятора)"""

    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.stopped = True
        self.fps = 30
        self.start_time = 0
        self.frame_count = 0
        self.use_fake_camera = False
        self.fake_frame_index = 0
        self.fake_movement = 0

    def start(self):
        """Запуск потока камеры"""
        if not self.stopped:
            return self

        if IN_ANDROID_EMULATOR:
            print("⚠ Android-эмулятор: используем имитацию камеры")
            self.use_fake_camera = True
            self.stopped = False
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._update_fake, args=(), daemon=True)
            self.thread.start()
            print("✓ Имитация камеры запущена")
            return self

        try:
            if platform == 'win':
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.src)
            else:
                self.cap = cv2.VideoCapture(self.src)

            if not self.cap or not self.cap.isOpened():
                print("⚠ Не удалось открыть камеру, переключаемся на имитацию")
                self.use_fake_camera = True
                self.stopped = False
                self.start_time = time.time()
                self.thread = threading.Thread(target=self._update_fake, args=(), daemon=True)
                self.thread.start()
                print("✓ Имитация камеры запущена")
                return self

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.stopped = False
            self.start_time = time.time()

            self.thread = threading.Thread(target=self._update, args=(), daemon=True)
            self.thread.start()
            print("✓ Реальная камера запущена")

        except Exception as e:
            print(f"✗ Ошибка запуска камеры: {e}")
            print("⚠ Переключаемся на имитацию камеры")
            self.use_fake_camera = True
            self.stopped = False
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._update_fake, args=(), daemon=True)
            self.thread.start()

        return self

    def _update(self):
        """Обновление кадров из реальной камеры"""
        while not self.stopped and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
                    self.frame_count += 1
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Ошибка чтения кадра: {e}")
                time.sleep(0.01)

    def _update_fake(self):
        """Создание фейковых кадров для имитации камеры"""
        while not self.stopped:
            try:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                self.fake_frame_index += 1
                self.fake_movement = int(50 * np.sin(self.fake_frame_index / 10.0))
                phase = self.fake_frame_index / 20.0

                # Голова
                head_y = 150 + int(20 * np.sin(phase))
                cv2.circle(frame, (320, head_y), 30, (100, 150, 200), -1)

                # Глаза
                cv2.circle(frame, (300, head_y - 10), 5, (255, 255, 255), -1)
                cv2.circle(frame, (340, head_y - 10), 5, (255, 255, 255), -1)
                cv2.circle(frame, (300, head_y - 10), 2, (0, 0, 0), -1)
                cv2.circle(frame, (340, head_y - 10), 2, (0, 0, 0), -1)

                # Тело
                body_y = 200 + int(30 * np.sin(phase))
                cv2.line(frame, (320, head_y + 20), (320, body_y), (150, 150, 150), 20)

                # Руки
                arm_angle = np.sin(phase * 2) * 0.8
                left_arm_x = 320 - int(60 * abs(np.cos(arm_angle)))
                right_arm_x = 320 + int(60 * abs(np.cos(arm_angle)))
                cv2.line(frame, (320, head_y + 30), (left_arm_x, body_y - 20), (150, 150, 150), 10)
                cv2.line(frame, (320, head_y + 30), (right_arm_x, body_y - 20), (150, 150, 150), 10)

                # Ноги
                leg_angle = np.sin(phase * 2 + 1) * 0.5
                left_leg_x = 320 - int(40 * abs(np.cos(leg_angle)))
                right_leg_x = 320 + int(40 * abs(np.cos(leg_angle)))
                cv2.line(frame, (320, body_y), (left_leg_x, 350), (150, 150, 150), 15)
                cv2.line(frame, (320, body_y), (right_leg_x, 350), (150, 150, 150), 15)

                # Текст
                cv2.putText(frame, "ДЕМО РЕЖИМ", (200, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Положение
                if abs(np.sin(phase)) > 0.7:
                    position = "ВНИЗУ"
                    color = (0, 0, 255)
                else:
                    position = "ВВЕРХУ"
                    color = (0, 255, 0)

                cv2.putText(frame, f"Положение: {position}", (200, 440),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                self.ret = True
                self.frame = frame
                self.frame_count += 1

                time.sleep(1.0 / 30.0)

            except Exception as e:
                print(f"Ошибка создания фейкового кадра: {e}")
                time.sleep(0.1)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.cap = None
        print("✓ Камера остановлена")


class ExerciseClassifier:
    """Класс для классификации типа упражнения по видеокадру"""

    def __init__(self):
        self.classes = ['отжимания', 'подтягивания', 'пресс', 'приседания']

    def classify(self, frame):
        """Определяет тип упражнения по кадру"""
        try:
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            height, width = gray.shape
            aspect_ratio = height / width

            # Анализ градиентов
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            horizontal_strength = np.mean(np.abs(sobelx))
            vertical_strength = np.mean(np.abs(sobely))
            gradient_ratio = horizontal_strength / (vertical_strength + 1e-6)

            # Распределение яркости по зонам
            top = np.mean(gray[:height // 3, :])
            middle = np.mean(gray[height // 3:2 * height // 3, :])
            bottom = np.mean(gray[2 * height // 3:, :])

            # Детекция кожи
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percent = np.mean(skin_mask) / 255

            # Логика определения
            scores = {}

            # Отжимания: горизонтальное изображение, сильные горизонтальные градиенты
            scores['отжимания'] = 0
            if aspect_ratio < 1.2:
                scores['отжимания'] += 0.3
            if gradient_ratio > 1.2:
                scores['отжимания'] += 0.3
            if skin_percent > 0.3:  # Много кожи (руки, грудь)
                scores['отжимания'] += 0.2
            if middle < top and middle < bottom:  # Середина темнее
                scores['отжимания'] += 0.2

            # Подтягивания: вертикальное изображение, сильные вертикальные градиенты
            scores['подтягивания'] = 0
            if aspect_ratio > 1.6:
                scores['подтягивания'] += 0.3
            if gradient_ratio < 0.8:
                scores['подтягивания'] += 0.3
            if skin_percent < 0.2:  # Мало кожи
                scores['подтягивания'] += 0.2
            if top < bottom:  # Верх темнее низа
                scores['подтягивания'] += 0.2

            # Пресс: горизонтальное, фокус на верхней части
            scores['пресс'] = 0
            if aspect_ratio < 1.3:
                scores['пресс'] += 0.2
            edges = cv2.Canny(gray, 50, 150)
            top_edges = np.mean(edges[:height // 2, :]) / 255
            if top_edges > 0.1:
                scores['пресс'] += 0.4
            if top > bottom:  # Верх ярче низа
                scores['пресс'] += 0.2

            # Приседания: средний аспект, темная середина
            scores['приседания'] = 0
            if 1.2 < aspect_ratio < 1.6:
                scores['приседания'] += 0.3
            if middle < top and middle < bottom:
                scores['приседания'] += 0.4
            if skin_percent > 0.25:  # Кожа на ногах
                scores['приседания'] += 0.3

            # Нормализация
            total = sum(scores.values())
            if total > 0:
                for k in scores:
                    scores[k] /= total

            # Определяем лучший класс
            best_class = max(scores, key=scores.get)
            confidence = scores[best_class]

            return best_class, confidence, scores

        except Exception as e:
            print(f"Ошибка классификации: {e}")
            return "неизвестно", 0, {}


class ExerciseAI:
    """Класс для работы с нейросетями упражнений"""

    def __init__(self):
        if getattr(sys, 'frozen', False):
            self.app_dir = os.path.dirname(sys.executable)
        else:
            self.app_dir = os.path.dirname(os.path.abspath(__file__))

        print(f"\n=== ПРОВЕРКА ПУТЕЙ К МОДЕЛЯМ ===")

        # ПУТИ К МОДЕЛЯМ - обновлено с вашими путями
        self.PUSHUP_MODEL_PATH = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\push_up_model.keras"
        self.PULLUP_MODEL_PATH = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\pull_up_model.keras"
        self.PRESS_MODEL_PATH = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\press_model.keras"
        self.SQUAT_MODEL_PATH = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\squat_model.keras"

        # ПУТИ К SCALER'АМ
        self.pushup_scaler_path = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\push_up_model_scaler.pkl"
        self.pullup_scaler_path = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\pull_up_model_scaler.pkl"
        self.press_scaler_path = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\press_model_scaler.pkl"
        self.squat_scaler_path = r"C:\Users\Sanya\Desktop\Mobile_trainer-main\squat_model_scaler.pkl"

        print(f"Отжимания модель: {os.path.exists(self.PUSHUP_MODEL_PATH)}")
        print(f"Подтягивания модель: {os.path.exists(self.PULLUP_MODEL_PATH)}")
        print(f"Пресс модель: {os.path.exists(self.PRESS_MODEL_PATH)}")
        print(f"Приседания модель: {os.path.exists(self.SQUAT_MODEL_PATH)}")

        # Загружаем модели
        self.pushup_model = None
        self.pullup_model = None
        self.press_model = None
        self.squat_model = None

        # Загружаем scaler'ы
        self.pushup_scaler = None
        self.pullup_scaler = None
        self.press_scaler = None
        self.squat_scaler = None

        # Счетчики повторений
        self.pushup_count = 0
        self.pullup_count = 0
        self.press_count = 0
        self.squat_count = 0

        # Состояния для детекции повторений
        self.pushup_state = "up"
        self.pullup_state = "down"
        self.press_state = "down"
        self.squat_state = "up"

        self.demo_mode = not TF_AVAILABLE or IN_ANDROID_EMULATOR
        self.models_loaded = False

        # Классификатор упражнений
        self.classifier = ExerciseClassifier()

        self.model_statuses = {
            "отжимания": "⏳ Загрузка...",
            "подтягивания": "⏳ Загрузка...",
            "пресс": "⏳ Загрузка...",
            "приседания": "⏳ Загрузка..."
        }

        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        """Загрузка всех моделей (в фоновом потоке)"""
        print("\n" + "=" * 60)
        print("ЗАГРУЗКА МОДЕЛЕЙ ИИ")
        print("=" * 60)

        if not TF_AVAILABLE:
            self.demo_mode = True
            for ex in self.model_statuses:
                self.model_statuses[ex] = "❌ TensorFlow не установлен"
            self.models_loaded = True
            return

        # Загружаем модель отжиманий (push_up_model.keras)
        if os.path.exists(self.PUSHUP_MODEL_PATH):
            try:
                # Пробуем загрузить с compile=False сначала
                self.pushup_model = keras.models.load_model(self.PUSHUP_MODEL_PATH, compile=False)
                print("✓ Модель отжиманий загружена (push_up_model.keras)")

                if os.path.exists(self.pushup_scaler_path):
                    self.pushup_scaler = joblib.load(self.pushup_scaler_path)
                    print("✓ Scaler отжиманий загружен")
                    self.model_statuses["отжимания"] = "✅ ИИ активен"
                else:
                    print("⚠ Scaler отжиманий не найден")
                    self.model_statuses["отжимания"] = "⚠ Нет scaler (демо-режим)"
                    self.pushup_model = None
            except Exception as e:
                print(f"✗ Ошибка загрузки модели отжиманий: {e}")
                self.model_statuses["отжимания"] = "❌ Ошибка загрузки"
                self.pushup_model = None
        else:
            print(f"✗ Модель отжиманий не найдена по пути: {self.PUSHUP_MODEL_PATH}")
            self.model_statuses["отжимания"] = "❌ Файл не найден"

        # Загружаем модель подтягиваний (pull_up_model.keras)
        if os.path.exists(self.PULLUP_MODEL_PATH):
            try:
                self.pullup_model = keras.models.load_model(self.PULLUP_MODEL_PATH, compile=False)
                print("✓ Модель подтягиваний загружена (pull_up_model.keras)")

                if os.path.exists(self.pullup_scaler_path):
                    self.pullup_scaler = joblib.load(self.pullup_scaler_path)
                    print("✓ Scaler подтягиваний загружен")
                    self.model_statuses["подтягивания"] = "✅ ИИ активен"
                else:
                    print("⚠ Scaler подтягиваний не найден")
                    self.model_statuses["подтягивания"] = "⚠ Нет scaler (демо-режим)"
                    self.pullup_model = None
            except Exception as e:
                print(f"✗ Ошибка загрузки модели подтягиваний: {e}")
                self.model_statuses["подтягивания"] = "❌ Ошибка загрузки"
                self.pullup_model = None
        else:
            print(f"✗ Модель подтягиваний не найдена по пути: {self.PULLUP_MODEL_PATH}")
            self.model_statuses["подтягивания"] = "❌ Файл не найден"

        # Загружаем модель пресса (press_model.keras) с обработкой ошибок версий
        if os.path.exists(self.PRESS_MODEL_PATH):
            try:
                # Пробуем загрузить с custom_objects для совместимости
                custom_objects = {'InputLayer': keras.layers.InputLayer}
                self.press_model = keras.models.load_model(
                    self.PRESS_MODEL_PATH,
                    compile=False,
                    custom_objects=custom_objects
                )
                print("✓ Модель пресса загружена (press_model.keras)")

                if os.path.exists(self.press_scaler_path):
                    self.press_scaler = joblib.load(self.press_scaler_path)
                    print("✓ Scaler пресса загружен")
                    self.model_statuses["пресс"] = "✅ ИИ активен"
                else:
                    print("⚠ Scaler пресса не найден")
                    self.model_statuses["пресс"] = "⚠ Нет scaler (демо-режим)"
                    self.press_model = None
            except Exception as e:
                print(f"✗ Ошибка загрузки модели пресса: {e}")
                print("  Пытаюсь загрузить с пользовательскими настройками...")

                # Вторая попытка с пользовательской загрузкой
                try:
                    import json
                    with open(self.PRESS_MODEL_PATH.replace('.keras', '_config.json'), 'r') as f:
                        config = json.load(f)
                    self.press_model = keras.Sequential.from_config(config)
                    print("✓ Модель пресса загружена из конфига")

                    if os.path.exists(self.press_scaler_path):
                        self.press_scaler = joblib.load(self.press_scaler_path)
                        self.model_statuses["пресс"] = "✅ ИИ активен"
                    else:
                        self.model_statuses["пресс"] = "⚠ Нет scaler"
                except:
                    self.model_statuses["пресс"] = "❌ Ошибка загрузки"
                    self.press_model = None
        else:
            print(f"✗ Модель пресса не найдена по пути: {self.PRESS_MODEL_PATH}")
            self.model_statuses["пресс"] = "❌ Файл не найден"

        # Загружаем модель приседаний (squat_model.keras) с обработкой ошибок версий
        if os.path.exists(self.SQUAT_MODEL_PATH):
            try:
                # Пробуем загрузить с custom_objects для совместимости
                custom_objects = {'InputLayer': keras.layers.InputLayer}
                self.squat_model = keras.models.load_model(
                    self.SQUAT_MODEL_PATH,
                    compile=False,
                    custom_objects=custom_objects
                )
                print("✓ Модель приседаний загружена (squat_model.keras)")

                if os.path.exists(self.squat_scaler_path):
                    self.squat_scaler = joblib.load(self.squat_scaler_path)
                    print("✓ Scaler приседаний загружен")
                    self.model_statuses["приседания"] = "✅ ИИ активен"
                else:
                    print("⚠ Scaler приседаний не найден")
                    self.model_statuses["приседания"] = "⚠ Нет scaler (демо-режим)"
                    self.squat_model = None
            except Exception as e:
                print(f"✗ Ошибка загрузки модели приседаний: {e}")
                print("  Пытаюсь загрузить с пользовательскими настройками...")

                # Вторая попытка с пользовательской загрузкой
                try:
                    import json
                    with open(self.SQUAT_MODEL_PATH.replace('.keras', '_config.json'), 'r') as f:
                        config = json.load(f)
                    self.squat_model = keras.Sequential.from_config(config)
                    print("✓ Модель приседаний загружена из конфига")

                    if os.path.exists(self.squat_scaler_path):
                        self.squat_scaler = joblib.load(self.squat_scaler_path)
                        self.model_statuses["приседания"] = "✅ ИИ активен"
                    else:
                        self.model_statuses["приседания"] = "⚠ Нет scaler"
                except:
                    self.model_statuses["приседания"] = "❌ Ошибка загрузки"
                    self.squat_model = None
        else:
            print(f"✗ Модель приседаний не найдена по пути: {self.SQUAT_MODEL_PATH}")
            self.model_statuses["приседания"] = "❌ Файл не найден"

        self.models_loaded = True
        self.demo_mode = not (self.pushup_model or self.pullup_model or self.press_model or self.squat_model)

        print("\n" + "=" * 60)
        print("СТАТУС МОДЕЛЕЙ:")
        print(f"Отжимания: {'✅' if self.pushup_model else '❌'}")
        print(f"Подтягивания: {'✅' if self.pullup_model else '❌'}")
        print(f"Пресс: {'✅' if self.press_model else '❌'}")
        print(f"Приседания: {'✅' if self.squat_model else '❌'}")
        print(f"Режим: {'ДЕМО' if self.demo_mode else 'ИИ'}")
        print("=" * 60)

    def convert_model_for_compatibility(self, model_path, output_path=None):
        """Конвертирует модель для совместимости с текущей версией TensorFlow"""
        try:
            # Загружаем модель
            model = keras.models.load_model(model_path, compile=False)

            # Сохраняем в новом формате
            if output_path is None:
                output_path = model_path.replace('.keras', '_converted.keras')

            model.save(output_path)
            print(f"✅ Модель сконвертирована и сохранена в: {output_path}")

            # Также сохраняем конфиг
            config = model.get_config()
            config_path = output_path.replace('.keras', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            return True
        except Exception as e:
            print(f"❌ Ошибка конвертации: {e}")
            return False

    def check_model_available(self, exercise):
        """Проверка доступности модели для упражнения"""
        if self.demo_mode:
            return False

        if exercise == "отжимания":
            return self.pushup_model is not None
        elif exercise == "подтягивания":
            return self.pullup_model is not None
        elif exercise == "пресс":
            return self.press_model is not None
        elif exercise == "приседания":
            return self.squat_model is not None
        return False

    def extract_pushup_features(self, frame):
        """Извлечение признаков для отжиманий (20 признаков)"""
        try:
            frame = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            features = []

            # 1. Яркостные характеристики
            features.append(np.mean(gray))
            features.append(np.std(gray))
            features.append(np.median(gray))

            # 2. Цветовые характеристики (кожа)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percent = np.mean(skin_mask) / 255
            features.append(skin_percent)

            # 3. Текстура через градиенты
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            features.append(np.mean(np.abs(sobelx)))
            features.append(np.mean(np.abs(sobely)))
            features.append(np.std(sobelx))
            features.append(np.std(sobely))

            # 4. Гистограмма ориентации градиентов
            magnitude, angle = cv2.cartToPolar(sobelx, sobely)
            hist, _ = np.histogram(angle, bins=4, range=(0, 2 * np.pi))
            hist = hist / (hist.sum() + 1e-6)
            features.extend(hist)

            # 5. Особенности контуров
            edges = cv2.Canny(gray, 50, 150)
            features.append(np.mean(edges) / 255)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                features.append(area / (perimeter + 1e-6))
            else:
                features.append(0)

            # Дополняем до 20 признаков
            while len(features) < 20:
                features.append(0)

            return np.array(features[:20])

        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return np.zeros(20)

    def extract_pullup_features(self, frame):
        """Извлечение признаков для подтягиваний (18 признаков)"""
        try:
            frame = cv2.resize(frame, (320, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            features = []

            height, width = gray.shape
            features.append(height / width)

            top_zone = gray[:height // 3, :]
            middle_zone = gray[height // 3:2 * height // 3, :]
            bottom_zone = gray[2 * height // 3:, :]

            features.append(np.mean(top_zone))
            features.append(np.mean(middle_zone))
            features.append(np.mean(bottom_zone))
            features.append(np.mean(top_zone) - np.mean(bottom_zone))

            edges = cv2.Canny(gray, 50, 150)
            top_edges = edges[:height // 4, :]
            features.append(np.mean(top_edges) / 255)

            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            features.append(np.mean(sobely))
            features.append(np.std(sobely))

            # ИСПРАВЛЕНИЕ: Определяем skin_mask перед использованием
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            features.append(np.mean(skin_mask) / 255)

            left_half = gray[:, :width // 2]
            right_half = gray[:, width // 2:]
            if left_half.shape == right_half.shape:
                vertical_symmetry = np.mean(np.abs(left_half - right_half))
            else:
                vertical_symmetry = 0
            features.append(vertical_symmetry)

            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            top_binary = thresh[:height // 3, :]
            middle_binary = thresh[height // 3:2 * height // 3, :]
            bottom_binary = thresh[2 * height // 3:, :]

            features.append(np.mean(top_binary) / 255)
            features.append(np.mean(middle_binary) / 255)
            features.append(np.mean(bottom_binary) / 255)

            features.append(np.max(gray) - np.min(gray))

            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            features.append(float(entropy))

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            vertical_ratio = np.mean(np.abs(sobely)) / (np.mean(np.abs(sobelx)) + 1e-6)
            features.append(vertical_ratio)

            while len(features) < 18:
                features.append(0)

            return np.array(features[:18])

        except Exception as e:
            print(f"Ошибка извлечения признаков для подтягиваний: {e}")
            return np.zeros(18)

    def extract_press_features(self, frame):
        """Извлечение признаков для пресса (20 признаков)"""
        try:
            frame = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            features = []

            features.append(np.mean(gray))
            features.append(np.std(gray))
            features.append(np.median(gray))

            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percent = np.mean(skin_mask) / 255
            features.append(skin_percent)

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            features.append(np.mean(np.abs(sobelx)))
            features.append(np.mean(np.abs(sobely)))
            features.append(np.std(sobelx))
            features.append(np.std(sobely))

            magnitude, angle = cv2.cartToPolar(sobelx, sobely)
            hist, _ = np.histogram(angle, bins=4, range=(0, 2 * np.pi))
            hist = hist / (hist.sum() + 1e-6)
            features.extend(hist)

            edges = cv2.Canny(gray, 50, 150)
            features.append(np.mean(edges) / 255)

            height, width = gray.shape
            top_half = edges[:height // 2, :]
            bottom_half = edges[height // 2:, :]
            features.append(np.mean(top_half) / 255)
            features.append(np.mean(bottom_half) / 255)

            while len(features) < 20:
                features.append(0)

            return np.array(features[:20])

        except Exception as e:
            print(f"Ошибка извлечения признаков для пресса: {e}")
            return np.zeros(20)

    def extract_squat_features(self, frame):
        """Извлечение признаков для приседаний (20 признаков)"""
        try:
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            features = []

            height, width = gray.shape
            features.append(height / width)
            features.append(np.mean(gray))
            features.append(np.std(gray))

            left_zone = gray[:, :width // 3]
            center_zone = gray[:, width // 3:2 * width // 3]
            right_zone = gray[:, 2 * width // 3:]

            features.append(np.mean(left_zone))
            features.append(np.mean(center_zone))
            features.append(np.mean(right_zone))

            top_zone = gray[:height // 3, :]
            middle_zone = gray[height // 3:2 * height // 3, :]
            bottom_zone = gray[2 * height // 3:, :]

            features.append(np.mean(top_zone))
            features.append(np.mean(middle_zone))
            features.append(np.mean(bottom_zone))

            features.append(np.mean(top_zone) / (np.mean(bottom_zone) + 1e-6))
            features.append(np.mean(center_zone) / (np.mean(left_zone) + 1e-6))
            features.append(np.mean(center_zone) / (np.mean(right_zone) + 1e-6))

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

            features.append(np.mean(np.abs(sobelx)))
            features.append(np.std(np.abs(sobelx)))
            features.append(np.mean(np.abs(sobely)))
            features.append(np.std(np.abs(sobely)))

            horizontal_energy = np.mean(np.abs(sobelx))
            vertical_energy = np.mean(np.abs(sobely))
            features.append(horizontal_energy / (vertical_energy + 1e-6))

            while len(features) < 20:
                features.append(0)

            return np.array(features[:20])

        except Exception as e:
            print(f"Ошибка извлечения признаков для приседаний: {e}")
            return np.zeros(20)

    def process_video_features(self, video_path, extract_func, max_frames=30):
        """Общая функция обработки видео для извлечения признаков"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None

        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = range(max(1, total_frames))

        all_features = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                features = extract_func(frame)
                all_features.append(features)

        cap.release()

        if len(all_features) > 0:
            features_array = np.array(all_features)
            mean_f = np.mean(features_array, axis=0)
            std_f = np.std(features_array, axis=0)
            max_f = np.max(features_array, axis=0)
            min_f = np.min(features_array, axis=0)
            combined = np.concatenate([mean_f, std_f, max_f, min_f])
            return combined

        return None

    def analyze_pushup_frame(self, frame):
        """Анализ кадра для отжиманий с использованием модели"""
        if self.check_model_available("отжимания"):
            try:
                # Для реального времени используем один кадр
                features = self.extract_pushup_features(frame)
                features_array = np.array([features])
                mean_features = np.mean(features_array, axis=0)
                std_features = np.std(features_array, axis=0)
                max_features = np.max(features_array, axis=0)
                min_features = np.min(features_array, axis=0)

                combined_features = np.concatenate([
                    mean_features, std_features, max_features, min_features
                ])

                if len(combined_features) != 80:
                    if len(combined_features) < 80:
                        combined_features = np.pad(combined_features, (0, 80 - len(combined_features)))
                    else:
                        combined_features = combined_features[:80]

                if self.pushup_scaler is not None:
                    features_scaled = self.pushup_scaler.transform([combined_features])
                else:
                    features_scaled = [combined_features]

                prediction = self.pushup_model.predict(features_scaled, verbose=0)
                probability = float(prediction[0][0])

                return {
                    'prediction': 'down' if probability > 0.5 else 'up',
                    'confidence': probability if probability > 0.5 else 1 - probability,
                    'probability': probability,
                    'demo': False
                }
            except Exception as e:
                print(f"Ошибка при использовании модели отжиманий: {e}")

        # Демо-режим
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        probability = 1.0 - (mean_brightness / 255.0)
        probability = np.clip(probability, 0.2, 0.9)

        return {
            'prediction': 'down' if probability > 0.6 else 'up',
            'confidence': abs(probability - 0.5) * 2,
            'probability': probability,
            'demo': True
        }

    def analyze_pullup_frame(self, frame):
        """Анализ кадра для подтягиваний с использованием модели"""
        if self.check_model_available("подтягивания"):
            try:
                features = self.extract_pullup_features(frame)
                features_array = np.array([features])
                mean_features = np.mean(features_array, axis=0)
                std_features = np.std(features_array, axis=0)
                max_features = np.max(features_array, axis=0)
                min_features = np.min(features_array, axis=0)

                combined_features = np.concatenate([
                    mean_features, std_features, max_features, min_features
                ])

                if len(combined_features) != 72:
                    if len(combined_features) < 72:
                        combined_features = np.pad(combined_features, (0, 72 - len(combined_features)))
                    else:
                        combined_features = combined_features[:72]

                if self.pullup_scaler is not None:
                    features_scaled = self.pullup_scaler.transform([combined_features])
                else:
                    features_scaled = [combined_features]

                prediction = self.pullup_model.predict(features_scaled, verbose=0)
                probability = float(prediction[0][0])

                return {
                    'prediction': 'up' if probability > 0.5 else 'down',
                    'confidence': probability if probability > 0.5 else 1 - probability,
                    'probability': probability,
                    'demo': False
                }
            except Exception as e:
                print(f"Ошибка при использовании модели подтягиваний: {e}")

        # Демо-режим
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (frame.shape[0] * frame.shape[1] + 1)
        probability = np.clip(edge_density * 3, 0.2, 0.9)

        return {
            'prediction': 'up' if probability > 0.6 else 'down',
            'confidence': abs(probability - 0.5) * 2,
            'probability': probability,
            'demo': True
        }

    def analyze_press_frame(self, frame):
        """Анализ кадра для пресса с использованием модели"""
        if self.check_model_available("пресс"):
            try:
                features = self.extract_press_features(frame)
                features_array = np.array([features])
                mean_features = np.mean(features_array, axis=0)
                std_features = np.std(features_array, axis=0)
                max_features = np.max(features_array, axis=0)
                min_features = np.min(features_array, axis=0)

                combined_features = np.concatenate([
                    mean_features, std_features, max_features, min_features
                ])

                if len(combined_features) != 80:
                    if len(combined_features) < 80:
                        combined_features = np.pad(combined_features, (0, 80 - len(combined_features)))
                    else:
                        combined_features = combined_features[:80]

                if self.press_scaler is not None:
                    features_scaled = self.press_scaler.transform([combined_features])
                else:
                    features_scaled = [combined_features]

                prediction = self.press_model.predict(features_scaled, verbose=0)
                probability = float(prediction[0][0])

                return {
                    'prediction': 'up' if probability > 0.5 else 'down',
                    'confidence': probability if probability > 0.5 else 1 - probability,
                    'probability': probability,
                    'demo': False
                }
            except Exception as e:
                print(f"Ошибка при использовании модели пресса: {e}")

        # Демо-режим
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        edges = cv2.Canny(gray, 50, 150)
        lower_half = edges[height // 2:, :]
        edge_density = np.sum(lower_half) / (lower_half.shape[0] * lower_half.shape[1] + 1)
        probability = np.clip(edge_density * 4, 0.2, 0.9)

        return {
            'prediction': 'up' if probability > 0.6 else 'down',
            'confidence': abs(probability - 0.5) * 2,
            'probability': probability,
            'demo': True
        }

    def analyze_squat_frame(self, frame):
        """Анализ кадра для приседаний с использованием модели"""
        if self.check_model_available("приседания"):
            try:
                features = self.extract_squat_features(frame)
                features_array = np.array([features])
                mean_features = np.mean(features_array, axis=0)
                std_features = np.std(features_array, axis=0)
                max_features = np.max(features_array, axis=0)
                min_features = np.min(features_array, axis=0)

                combined_features = np.concatenate([
                    mean_features, std_features, max_features, min_features
                ])

                if len(combined_features) != 80:
                    if len(combined_features) < 80:
                        combined_features = np.pad(combined_features, (0, 80 - len(combined_features)))
                    else:
                        combined_features = combined_features[:80]

                if self.squat_scaler is not None:
                    features_scaled = self.squat_scaler.transform([combined_features])
                else:
                    features_scaled = [combined_features]

                prediction = self.squat_model.predict(features_scaled, verbose=0)
                probability = float(prediction[0][0])

                return {
                    'prediction': 'down' if probability > 0.5 else 'up',
                    'confidence': probability if probability > 0.5 else 1 - probability,
                    'probability': probability,
                    'demo': False
                }
            except Exception as e:
                print(f"Ошибка при использовании модели приседаний: {e}")

        # Демо-режим
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        edges = cv2.Canny(gray, 50, 150)
        lower_half = edges[height // 2:, :]
        edge_density = np.sum(lower_half) / (lower_half.shape[0] * lower_half.shape[1] + 1)
        probability = np.clip(edge_density * 3, 0.2, 0.9)

        return {
            'prediction': 'down' if probability > 0.6 else 'up',
            'confidence': abs(probability - 0.5) * 2,
            'probability': probability,
            'demo': True
        }

    def detect_pushup_rep(self, prediction):
        """Детекция повторения отжимания"""
        if prediction is None:
            return False

        if prediction['prediction'] == 'down' and self.pushup_state == "up":
            self.pushup_state = "down"
            return False
        elif prediction['prediction'] == 'up' and self.pushup_state == "down":
            self.pushup_state = "up"
            self.pushup_count += 1
            print(f"✓ Отжимание! Счет: {self.pushup_count}")
            return True
        return False

    def detect_pullup_rep(self, prediction):
        """Детекция повторения подтягивания"""
        if prediction is None:
            return False

        if prediction['prediction'] == 'up' and self.pullup_state == "down":
            self.pullup_state = "up"
            return False
        elif prediction['prediction'] == 'down' and self.pullup_state == "up":
            self.pullup_state = "down"
            self.pullup_count += 1
            print(f"✓ Подтягивание! Счет: {self.pullup_count}")
            return True
        return False

    def detect_press_rep(self, prediction):
        """Детекция повторения пресса"""
        if prediction is None:
            return False

        if prediction['prediction'] == 'up' and self.press_state == "down":
            self.press_state = "up"
            return False
        elif prediction['prediction'] == 'down' and self.press_state == "up":
            self.press_state = "down"
            self.press_count += 1
            print(f"✓ Пресс! Счет: {self.press_count}")
            return True
        return False

    def detect_squat_rep(self, prediction):
        """Детекция повторения приседаний"""
        if prediction is None:
            return False

        if prediction['prediction'] == 'down' and self.squat_state == "up":
            self.squat_state = "down"
            return False
        elif prediction['prediction'] == 'up' and self.squat_state == "down":
            self.squat_state = "up"
            self.squat_count += 1
            print(f"✓ Приседание! Счет: {self.squat_count}")
            return True
        return False

    def reset_counts(self):
        """Сброс счетчиков"""
        self.pushup_count = 0
        self.pullup_count = 0
        self.press_count = 0
        self.squat_count = 0
        self.pushup_state = "up"
        self.pullup_state = "down"
        self.press_state = "down"
        self.squat_state = "up"
        print("✓ Счетчики сброшены")

    def get_count(self, exercise):
        """Получить количество повторений для упражнения"""
        if exercise == "отжимания":
            return self.pushup_count
        elif exercise == "подтягивания":
            return self.pullup_count
        elif exercise == "пресс":
            return self.press_count
        elif exercise == "приседания":
            return self.squat_count
        return 0

    def get_model_status(self, exercise):
        """Получить статус модели для упражнения"""
        if not self.models_loaded:
            return "⏳ Загрузка..."
        return self.model_statuses.get(exercise, "ℹ️ Демо-режим")

    def verify_exercise(self, frame, expected_exercise):
        """Проверяет, соответствует ли кадр ожидаемому упражнению"""
        detected_exercise, confidence, scores = self.classifier.classify(frame)

        if detected_exercise == expected_exercise:
            return True, confidence, f"✓ Упражнение определено как {detected_exercise}"
        else:
            return False, confidence, f"⚠ Обнаружено {detected_exercise}, а выбрано {expected_exercise}"


class KivyCamera(Image):
    """Виджет для отображения видеопотока"""

    def __init__(self, exercise_ai, exercise_name, on_rep_callback=None, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.exercise_ai = exercise_ai
        self.exercise_name = exercise_name
        self.on_rep_callback = on_rep_callback
        self.camera_stream = None
        self.current_count = 0
        self.rep_cooldown = 0.5
        self.camera_available = False
        self.update_event = None
        self.rep_animation = 0
        self.verification_frames = 0
        self.misalignment_warning_shown = False

        Clock.schedule_once(self.start_camera, 0.1)

    def start_camera(self, dt=None):
        try:
            self.camera_stream = CameraStream(0).start()
            self.camera_available = True
            self.update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)
            print(f"✓ Видеопоток запущен для {self.exercise_name}")
        except Exception as e:
            self.camera_available = False
            print(f"✗ Ошибка запуска камеры: {e}")

    def update(self, dt):
        if not self.camera_available or self.camera_stream is None:
            return

        try:
            ret, frame = self.camera_stream.read()
            if not ret or frame is None:
                return

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)

            # Проверяем соответствие упражнению (каждые 30 кадров)
            self.verification_frames += 1
            if self.verification_frames % 30 == 0:
                is_correct, conf, msg = self.exercise_ai.verify_exercise(frame, self.exercise_name)
                if not is_correct and not self.misalignment_warning_shown and conf > 0.4:
                    self.misalignment_warning_shown = True
                    # Показываем предупреждение в UI
                    app = App.get_running_app()
                    app.show_warning_popup(msg)

            prediction = None
            is_rep = False

            if self.exercise_name == "отжимания":
                prediction = self.exercise_ai.analyze_pushup_frame(frame)
                is_rep = self.exercise_ai.detect_pushup_rep(prediction)
                self.current_count = self.exercise_ai.pushup_count
            elif self.exercise_name == "подтягивания":
                prediction = self.exercise_ai.analyze_pullup_frame(frame)
                is_rep = self.exercise_ai.detect_pullup_rep(prediction)
                self.current_count = self.exercise_ai.pullup_count
            elif self.exercise_name == "пресс":
                prediction = self.exercise_ai.analyze_press_frame(frame)
                is_rep = self.exercise_ai.detect_press_rep(prediction)
                self.current_count = self.exercise_ai.press_count
            elif self.exercise_name == "приседания":
                prediction = self.exercise_ai.analyze_squat_frame(frame)
                is_rep = self.exercise_ai.detect_squat_rep(prediction)
                self.current_count = self.exercise_ai.squat_count

            if is_rep:
                self.rep_animation = 10
                self.misalignment_warning_shown = False  # Сбрасываем предупреждение при повторении
                if self.on_rep_callback:
                    self.on_rep_callback()

            # Отрисовка
            model_status = self.exercise_ai.get_model_status(self.exercise_name)
            cv2.putText(frame, f"Status: {model_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"СЧЕТ: {self.current_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

            if prediction:
                color = (0, 255, 0) if prediction['prediction'] == 'up' else (0, 0, 255)
                demo_text = "DEMO" if prediction.get('demo', False) else "AI"
                position_text = "ВВЕРХУ" if prediction['prediction'] == 'up' else "ВНИЗУ"
                cv2.putText(frame, f"{demo_text}: {position_text}", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if self.rep_animation > 0:
                cv2.putText(frame, "ПОВТОРЕНИЕ!", (200, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                self.rep_animation -= 1

            if self.misalignment_warning_shown:
                cv2.putText(frame, "⚠ НЕПРАВИЛЬНОЕ УПРАЖНЕНИЕ!", (150, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture

        except Exception as e:
            print(f"Ошибка обновления кадра: {e}")

    def stop_camera(self):
        if self.update_event:
            self.update_event.cancel()
        if self.camera_stream:
            self.camera_stream.stop()
        self.camera_stream = None


class ProfileScreen(BoxLayout):
    """Экран личного кабинета со статистикой"""

    def __init__(self, profile, on_back_callback, **kwargs):
        super(ProfileScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.padding = dp(20)

        self.profile = profile
        self.on_back = on_back_callback

        # Заголовок
        title = Label(
            text="👤 ЛИЧНЫЙ КАБИНЕТ",
            font_size='28sp',
            bold=True,
            color=(0.2, 0.8, 1, 1),
            size_hint=(1, 0.1),
            halign='center'
        )
        self.add_widget(title)

        # Основная статистика
        stats_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.25), spacing=dp(10))

        # Карточка с общей статистикой
        total_card = BoxLayout(orientation='vertical', size_hint=(1, 1), padding=dp(15))
        with total_card.canvas.before:
            Color(0.15, 0.15, 0.15, 1)
            self.total_rect = Rectangle(size=total_card.size, pos=total_card.pos)
        total_card.bind(pos=self.update_rect, size=self.update_rect)

        total_stats = (
            f"[color=FFD700]📅 Дней с начала:[/color] [b]{self.profile.total_days}[/b]\n"
            f"[color=FFD700]🔥 Текущая серия:[/color] [b]{self.profile.streak_days}[/b] дней\n"
            f"[color=FFD700]💪 Всего тренировок:[/color] [b]{self.profile.total_workouts}[/b]\n"
            f"[color=FFD700]🎯 Всего повторений:[/color] [b]{sum(self.profile.total_reps.values())}[/b]"
        )

        total_label = Label(
            text=total_stats,
            font_size='16sp',
            markup=True,
            halign='left',
            valign='middle',
            color=(1, 1, 1, 1)
        )
        total_card.add_widget(total_label)
        stats_layout.add_widget(total_card)

        self.add_widget(stats_layout)

        # Информация о текущих целях
        targets_title = Label(
            text="🎯 ТЕКУЩИЕ ЦЕЛИ",
            font_size='18sp',
            bold=True,
            color=(0.2, 0.8, 1, 1),
            size_hint=(1, 0.05),
            halign='center'
        )
        self.add_widget(targets_title)

        # Сетка с целями
        targets_grid = GridLayout(cols=2, spacing=dp(15), padding=dp(10), size_hint=(1, 0.25))

        exercises = [
            ("💪 Отжимания", "отжимания", (0.3, 0.6, 0.3, 0.8)),
            ("⬆️ Подтягивания", "подтягивания", (0.3, 0.3, 0.6, 0.8)),
            ("🔄 Пресс", "пресс", (0.6, 0.3, 0.3, 0.8)),
            ("🦵 Приседания", "приседания", (0.6, 0.6, 0.3, 0.8))
        ]

        for display_name, key, color in exercises:
            card = BoxLayout(orientation='vertical', padding=dp(10))
            with card.canvas.before:
                Color(*color)
                self.ex_rect = Rectangle(size=card.size, pos=card.pos)
            card.bind(pos=self.update_rect, size=self.update_rect)

            current = self.profile.current_targets.get(key, 0)
            original = self.profile.original_max.get(key, 0)
            progress = int((current / max(1, original)) * 100) if original > 0 else 0

            stats = (
                f"[b]{display_name}[/b]\n\n"
                f"[color=FFFFFF]🎯 Цель:[/color] [b]{current}[/b]\n"
                f"[color=AAAAAA]📊 Исх. максимум:[/color] {original}\n"
                f"[color=00FF00]📈 Прогресс:[/color] {progress}%"
            )

            label = Label(
                text=stats,
                markup=True,
                font_size='14sp',
                halign='center',
                valign='middle',
                color=(1, 1, 1, 1)
            )
            card.add_widget(label)
            targets_grid.add_widget(card)

        self.add_widget(targets_grid)

        # Статистика по упражнениям
        exercises_title = Label(
            text="📊 СТАТИСТИКА ПО УПРАЖНЕНИЯМ",
            font_size='18sp',
            bold=True,
            color=(0.2, 0.8, 1, 1),
            size_hint=(1, 0.05),
            halign='center'
        )
        self.add_widget(exercises_title)

        # Сетка с упражнениями
        grid = GridLayout(cols=2, spacing=dp(15), padding=dp(10), size_hint=(1, 0.25))

        for display_name, key, color in exercises:
            card = BoxLayout(orientation='vertical', padding=dp(10))
            with card.canvas.before:
                Color(*color)
                self.ex_rect = Rectangle(size=card.size, pos=card.pos)
            card.bind(pos=self.update_rect, size=self.update_rect)

            total = self.profile.total_reps.get(key, 0)
            max_reps = self.profile.max_reps.get(key, 0)

            stats = (
                f"[b]{display_name}[/b]\n\n"
                f"[color=FFFFFF]📊 Всего:[/color] [b]{total}[/b]\n"
                f"[color=FFD700]🏆 Макс:[/color] [b]{max_reps}[/b]"
            )

            label = Label(
                text=stats,
                markup=True,
                font_size='14sp',
                halign='center',
                valign='middle',
                color=(1, 1, 1, 1)
            )
            card.add_widget(label)
            grid.add_widget(card)

        self.add_widget(grid)

        # Кнопка возврата
        back_btn = Button(
            text="◀️ НАЗАД К ПРОГРЕССУ",
            font_size='18sp',
            size_hint=(1, 0.1),
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1)
        )
        back_btn.bind(on_press=self.go_back)
        self.add_widget(back_btn)

    def update_rect(self, instance, value):
        """Обновление позиции прямоугольника"""
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(0.15, 0.15, 0.15, 1)
            Rectangle(size=instance.size, pos=instance.pos)

    def go_back(self, instance):
        """Возврат к экрану прогресса"""
        self.on_back()


class MaxInputScreen(BoxLayout):
    """Экран ввода максимальных показателей (без бега)"""

    def __init__(self, profile, on_submit_callback, **kwargs):
        super(MaxInputScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(20)
        self.padding = dp(20)
        self.profile = profile
        self.on_submit = on_submit_callback

        # Проверяем, нужно ли обновить максимумы
        needs_update = profile.needs_max_update()

        if not needs_update:
            # Если не нужно обновлять, сразу вызываем колбэк с текущими значениями
            Clock.schedule_once(lambda dt: self.on_submit(profile.original_max), 0)
            return

        # Заголовок
        title = Label(
            text="🏋️ ОБНОВИТЕ ВАШ МАКСИМУМ 🏋️",
            font_size='24sp',
            bold=True,
            color=(0.2, 0.8, 1, 1),
            size_hint=(1, 0.15)
        )
        self.add_widget(title)

        info = Label(
            text="Прошла неделя! Укажите ваше новое максимальное количество повторений",
            font_size='16sp',
            halign='center',
            size_hint=(1, 0.1)
        )
        self.add_widget(info)

        # Поля ввода
        self.inputs = {}

        exercises = [
            ("отжимания", "💪 Отжимания", profile.original_max.get("отжимания", 50)),
            ("подтягивания", "⬆️ Подтягивания", profile.original_max.get("подтягивания", 10)),
            ("пресс", "🔄 Пресс", profile.original_max.get("пресс", 30)),
            ("приседания", "🦵 Приседания", profile.original_max.get("приседания", 50))
        ]

        for key, label_text, default in exercises:
            box = BoxLayout(orientation='horizontal', size_hint=(1, 0.12), spacing=dp(10))

            label = Label(
                text=label_text,
                font_size='16sp',
                size_hint=(0.4, 1),
                halign='left'
            )
            box.add_widget(label)

            text_input = TextInput(
                text=str(default),
                font_size='18sp',
                multiline=False,
                input_filter='int',
                size_hint=(0.3, 1)
            )
            box.add_widget(text_input)

            self.inputs[key] = text_input
            self.add_widget(box)

        # Кнопка подтверждения
        submit_btn = Button(
            text="✅ СОХРАНИТЬ МАКСИМУМ",
            font_size='20sp',
            background_color=(0.2, 0.8, 0.2, 1),
            size_hint=(1, 0.15)
        )
        submit_btn.bind(on_press=self.submit)
        self.add_widget(submit_btn)

    def submit(self, instance):
        """Обработка подтверждения"""
        try:
            max_values = {}
            for key, input_field in self.inputs.items():
                text = input_field.text.strip()
                if not text:
                    text = "0"
                max_values[key] = int(text)

            self.profile.update_max_values(max_values)
            self.on_submit(max_values)
        except ValueError as e:
            app = App.get_running_app()
            app.show_error_popup("Введите корректные числа!")


class ExerciseCard(BoxLayout):
    """Карточка упражнения с кнопкой увеличения"""

    def __init__(self, exercise_name, current_value, target_value, on_increase_callback, **kwargs):
        super(ExerciseCard, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = dp(160)
        self.padding = dp(10)
        self.spacing = dp(5)

        # Фон
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(pos=self.update_rect, size=self.update_rect)

        self.exercise_name = exercise_name
        self.current_value = current_value
        self.target_value = target_value
        self.on_increase = on_increase_callback

        # Название и текущее значение
        title_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.25))

        emoji_map = {
            "отжимания": "💪",
            "подтягивания": "⬆️",
            "пресс": "🔄",
            "приседания": "🦵"
        }

        title_label = Label(
            text=f"{emoji_map.get(exercise_name, '🏋️')} {exercise_name.capitalize()}",
            font_size='18sp',
            bold=True,
            halign='left',
            valign='middle',
            color=(1, 1, 1, 1),
            size_hint=(0.6, 1)
        )
        title_layout.add_widget(title_label)

        self.value_label = Label(
            text=f"[color=00FF00]{current_value}[/color]",
            font_size='20sp',
            bold=True,
            markup=True,
            halign='right',
            valign='middle',
            size_hint=(0.4, 1)
        )
        title_layout.add_widget(self.value_label)

        self.add_widget(title_layout)

        # Прогресс-бар
        progress_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=dp(5))

        progress_percent = min(100, int((current_value / max(1, target_value)) * 100))
        self.progress_bar = ProgressBar(max=target_value, value=current_value, size_hint=(0.7, 1))
        progress_layout.add_widget(self.progress_bar)

        self.percent_label = Label(
            text=f"[color=FFFF00]{progress_percent}%[/color]",
            font_size='14sp',
            markup=True,
            size_hint=(0.3, 1),
            halign='center',
            valign='middle'
        )
        progress_layout.add_widget(self.percent_label)

        self.add_widget(progress_layout)

        # Информация о цели
        target_label = Label(
            text=f"[color=AAAAAA]Цель:[/color] [color=FFFFFF]{target_value}[/color]",
            markup=True,
            font_size='14sp',
            size_hint=(1, 0.2),
            halign='center',
            valign='middle'
        )
        self.add_widget(target_label)

        # Кнопка увеличения
        increase_btn = Button(
            text="➕ УВЕЛИЧИТЬ ЦЕЛЬ",
            font_size='14sp',
            background_color=(0.3, 0.3, 0.8, 1),
            color=(1, 1, 1, 1),
            size_hint=(1, 0.35)
        )
        increase_btn.bind(on_press=self.increase)
        self.add_widget(increase_btn)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def increase(self, instance):
        """Увеличение значения цели"""
        if self.on_increase:
            self.on_increase(self.exercise_name, self.current_value, self.target_value)


class ProgressScreen(BoxLayout):
    """Экран отслеживания прогресса"""

    def __init__(self, max_values, exercise_ai, profile, **kwargs):
        super(ProgressScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.padding = dp(10)

        self.max_values = max_values.copy()
        self.current_values = max_values.copy()
        self.target_values = profile.current_targets.copy()
        self.exercise_ai = exercise_ai
        self.profile = profile

        # Верхняя панель с кнопкой профиля
        top_panel = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=dp(10))

        # Заголовок
        title = Label(
            text="📊 ВАШ ПРОГРЕСС",
            font_size='20sp',
            bold=True,
            color=(0.2, 0.8, 1, 1),
            size_hint=(0.4, 1),
            halign='left',
            valign='middle'
        )
        top_panel.add_widget(title)

        # Кнопка профиля
        profile_btn = Button(
            text="👤 ПРОФИЛЬ",
            font_size='14sp',
            background_color=(0.4, 0.4, 0.8, 1),
            color=(1, 1, 1, 1),
            size_hint=(0.3, 1)
        )
        profile_btn.bind(on_press=self.show_profile)
        top_panel.add_widget(profile_btn)

        # Кнопка сброса
        reset_btn = Button(
            text="🔄 Сброс",
            font_size='14sp',
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1),
            size_hint=(0.3, 1)
        )
        reset_btn.bind(on_press=self.reset_all)
        top_panel.add_widget(reset_btn)

        self.add_widget(top_panel)

        # Контейнер для карточек с прокруткой
        scroll = ScrollView(size_hint=(1, 0.5))
        self.cards_layout = BoxLayout(
            orientation='vertical',
            spacing=dp(10),
            size_hint_y=None
        )
        self.cards_layout.bind(minimum_height=self.cards_layout.setter('height'))

        # Создаем карточки для каждого упражнения
        self.cards = {}

        for exercise, max_val in max_values.items():
            card = ExerciseCard(
                exercise,
                self.current_values[exercise],
                self.target_values[exercise],
                self.on_increase_click
            )
            self.cards_layout.add_widget(card)
            self.cards[exercise] = card

        scroll.add_widget(self.cards_layout)
        self.add_widget(scroll)

        # Кнопка тренировки
        train_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=dp(10))

        self.train_btn = Button(
            text="🏋️ ТРЕНИРОВАТЬСЯ",
            font_size='18sp',
            background_color=(0.2, 0.8, 0.2, 1),
            color=(1, 1, 1, 1),
            size_hint=(0.7, 1)
        )
        self.train_btn.bind(on_press=self.start_training)
        train_layout.add_widget(self.train_btn)

        # Кнопка возврата
        back_btn = Button(
            text="◀️ Назад",
            font_size='16sp',
            background_color=(0.5, 0.5, 0.5, 1),
            color=(1, 1, 1, 1),
            size_hint=(0.3, 1)
        )
        back_btn.bind(on_press=self.go_back)
        train_layout.add_widget(back_btn)

        self.add_widget(train_layout)

        # Статус ИИ
        self.status_label = Label(
            text=self.get_ai_status_text(),
            font_size='14sp',
            color=(0.8, 0.8, 0.8, 1),
            size_hint=(1, 0.05),
            halign='center'
        )
        self.add_widget(self.status_label)

        # Информация о пользователе
        user_info = f"📅 День {self.profile.total_days}  |  🔥 Серия: {self.profile.streak_days} дней"
        self.user_label = Label(
            text=user_info,
            font_size='12sp',
            color=(0.8, 0.8, 0.8, 1),
            size_hint=(1, 0.05),
            halign='center'
        )
        self.add_widget(self.user_label)

        # Обновляем статус через 2 секунды
        Clock.schedule_once(lambda dt: self.update_status(), 2)

    def get_ai_status_text(self):
        """Получение текста статуса ИИ"""
        if self.exercise_ai.demo_mode:
            return "⚠ ДЕМО-РЕЖИМ: Счет считается по движению"
        else:
            return "✅ ИИ активен: анализ техники выполнения"

    def update_status(self):
        """Обновление статуса"""
        self.status_label.text = self.get_ai_status_text()

    def on_increase_click(self, exercise, current, target):
        """Обработка нажатия на кнопку увеличения"""
        # Увеличиваем цель в профиле
        if self.profile.increase_target(exercise):
            # Обновляем отображение
            self.target_values[exercise] = self.profile.current_targets[exercise]
            if exercise in self.cards:
                self.cards[exercise].target_value = self.target_values[exercise]
                self.cards[exercise].progress_bar.max = self.target_values[exercise]
                # Обновляем процент
                progress_percent = min(100, int((current / max(1, self.target_values[exercise])) * 100))
                self.cards[exercise].percent_label.text = f"[color=FFFF00]{progress_percent}%[/color]"
                self.cards[exercise].progress_bar.value = current

            app = App.get_running_app()
            app.show_success_popup(f"Цель для {exercise} увеличена!\nНовая цель: {self.target_values[exercise]}")

    def reset_all(self, instance):
        """Сброс всех значений"""
        for exercise in self.current_values:
            self.current_values[exercise] = self.max_values[exercise]
            if exercise in self.cards:
                self.cards[exercise].current_value = self.max_values[exercise]
                self.cards[exercise].value_label.text = f"[color=00FF00]{self.max_values[exercise]}[/color]"

                # Обновляем процент
                progress_percent = min(100,
                                       int((self.max_values[exercise] / max(1, self.target_values[exercise])) * 100))
                self.cards[exercise].percent_label.text = f"[color=FFFF00]{progress_percent}%[/color]"
                self.cards[exercise].progress_bar.value = self.max_values[exercise]

        print("✓ Все значения сброшены")

    def start_training(self, instance):
        """Запуск тренировки"""
        app = App.get_running_app()
        app.show_exercise_selection(self.current_values)

    def show_profile(self, instance):
        """Показать профиль пользователя"""
        app = App.get_running_app()
        app.show_profile_screen()

    def go_back(self, instance):
        """Возврат к вводу максимумов"""
        app = App.get_running_app()
        app.show_max_input()


class TrainingScreen(BoxLayout):
    """Экран выбора упражнения для тренировки"""

    def __init__(self, current_values, exercise_ai, **kwargs):
        super(TrainingScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.padding = dp(10)

        self.current_values = current_values
        self.exercise_ai = exercise_ai

        # Заголовок
        title = Label(
            text="🏋️ ВЫБЕРИТЕ УПРАЖНЕНИЕ",
            font_size='24sp',
            bold=True,
            color=(0.2, 0.8, 1, 1),
            size_hint=(1, 0.15)
        )
        self.add_widget(title)

        # Сетка упражнений
        grid = GridLayout(cols=2, spacing=dp(15), padding=dp(15), size_hint=(1, 0.7))

        exercises = [
            ("отжимания", "💪", (0.3, 0.6, 0.3, 1)),
            ("подтягивания", "⬆️", (0.3, 0.3, 0.6, 1)),
            ("пресс", "🔄", (0.6, 0.3, 0.3, 1)),
            ("приседания", "🦵", (0.6, 0.6, 0.3, 1))
        ]

        for exercise, emoji, color in exercises:
            if exercise in current_values:
                status = exercise_ai.get_model_status(exercise)
                btn_text = f"{emoji}\n{exercise.capitalize()}\nТекущий: {current_values[exercise]}\n{status}"

                btn = Button(
                    text=btn_text,
                    font_size='16sp',
                    size_hint=(1, 1),
                    background_color=color
                )
                btn.bind(on_press=lambda x, ex=exercise: self.start_exercise(ex))
                grid.add_widget(btn)

        self.add_widget(grid)

        # Кнопка возврата
        back_btn = Button(
            text="◀️ НАЗАД К ПРОГРЕССУ",
            font_size='18sp',
            size_hint=(1, 0.1),
            background_color=(0.5, 0.5, 0.5, 1)
        )
        back_btn.bind(on_press=self.go_back)
        self.add_widget(back_btn)

    def start_exercise(self, exercise):
        """Запуск упражнения"""
        app = App.get_running_app()
        app.start_exercise_session(exercise, self.current_values[exercise])

    def go_back(self, instance):
        """Возврат к прогрессу"""
        app = App.get_running_app()
        app.show_progress_screen()


class ExerciseSessionScreen(BoxLayout):
    """Экран выполнения упражнения"""

    def __init__(self, exercise_ai, exercise_name, target_value, **kwargs):
        super(ExerciseSessionScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(5)
        self.padding = dp(5)

        self.exercise_ai = exercise_ai
        self.exercise_name = exercise_name
        self.target_value = target_value
        self.start_count = exercise_ai.get_count(exercise_name)

        # Заголовок
        status = exercise_ai.get_model_status(exercise_name)
        title = Label(
            text=f"{exercise_name.capitalize()} - Цель: {target_value}\n{status}",
            font_size='18sp',
            bold=True,
            color=(0.2, 0.6, 1, 1),
            size_hint=(1, 0.1)
        )
        self.add_widget(title)

        # Видеопоток
        self.camera_widget = KivyCamera(
            exercise_ai,
            exercise_name,
            on_rep_callback=self.on_rep
        )
        self.add_widget(self.camera_widget)

        # Информация
        info_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=dp(10))

        self.count_label = Label(
            text=f"Текущий счет: 0",
            font_size='18sp',
            color=(1, 1, 1, 1)
        )
        info_layout.add_widget(self.count_label)

        self.status_label = Label(
            text="Выполняется...",
            font_size='16sp',
            color=(1, 1, 1, 1)
        )
        info_layout.add_widget(self.status_label)

        self.add_widget(info_layout)

        # Кнопки
        btn_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=dp(10))

        reset_btn = Button(
            text="🔄 Сброс",
            background_color=(0.3, 0.3, 0.8, 1),
            size_hint=(0.3, 1)
        )
        reset_btn.bind(on_press=self.reset_counter)
        btn_layout.add_widget(reset_btn)

        complete_btn = Button(
            text="✅ Завершить",
            background_color=(0.2, 0.8, 0.2, 1),
            size_hint=(0.4, 1)
        )
        complete_btn.bind(on_press=self.complete_exercise)
        btn_layout.add_widget(complete_btn)

        back_btn = Button(
            text="✖ Отмена",
            background_color=(0.8, 0.3, 0.3, 1),
            size_hint=(0.3, 1)
        )
        back_btn.bind(on_press=self.cancel_exercise)
        btn_layout.add_widget(back_btn)

        self.add_widget(btn_layout)

        # Планировщик обновления
        Clock.schedule_interval(self.update_info, 0.1)

    def on_rep(self):
        """Колбэк при повторении"""
        current = self.exercise_ai.get_count(self.exercise_name) - self.start_count
        if current >= self.target_value:
            self.status_label.text = "✅ Цель достигнута!"
            self.status_label.color = (0, 1, 0, 1)

    def update_info(self, dt):
        """Обновление информации"""
        current = self.exercise_ai.get_count(self.exercise_name) - self.start_count
        self.count_label.text = f"Сделано: {current}/{self.target_value}"

    def reset_counter(self, instance):
        """Сброс счетчика"""
        self.exercise_ai.reset_counts()
        self.start_count = 0
        self.status_label.text = "Счет сброшен"
        self.status_label.color = (1, 1, 0, 1)
        Clock.schedule_once(lambda dt: setattr(self.status_label, 'text', "Выполняется..."), 1)

    def complete_exercise(self, instance):
        """Завершение упражнения с сохранением"""
        current = self.exercise_ai.get_count(self.exercise_name) - self.start_count
        app = App.get_running_app()

        if current > 0:
            # Увеличиваем значение в прогрессе
            app.update_exercise_value(self.exercise_name, current)
            # Сохраняем в профиль
            app.add_workout_to_profile(self.exercise_name, current)
            app.show_success_popup(f"Вы сделали {current} {self.exercise_name}!")

        self.cleanup()
        app.show_training_screen()

    def cancel_exercise(self, instance):
        """Отмена упражнения без сохранения"""
        self.cleanup()
        app = App.get_running_app()
        app.show_training_screen()

    def cleanup(self):
        """Очистка"""
        self.camera_widget.stop_camera()
        Clock.unschedule(self.update_info)
        self.exercise_ai.reset_counts()


class Mobile_Trainer(App):
    def build(self):
        self.title = "Mobile Trainer PRO"
        Window.clearcolor = (0.1, 0.1, 0.1, 1)

        self.exercise_ai = ExerciseAI()
        self.profile = UserProfile()
        self.current_values = {}
        self.layout = BoxLayout()

        # Сразу показываем экран ввода максимумов
        self.show_max_input()

        return self.layout

    def show_max_input(self):
        """Показать экран ввода максимумов"""
        self.layout.clear_widgets()
        self.layout.add_widget(MaxInputScreen(self.profile, self.on_max_submit))

    def on_max_submit(self, max_values):
        """Обработка ввода максимумов"""
        self.current_values = max_values
        self.show_progress_screen()

    def show_progress_screen(self):
        """Показать экран прогресса"""
        self.layout.clear_widgets()
        self.layout.add_widget(ProgressScreen(self.current_values, self.exercise_ai, self.profile))

    def show_training_screen(self):
        """Показать экран выбора тренировки"""
        self.layout.clear_widgets()
        self.layout.add_widget(TrainingScreen(self.current_values, self.exercise_ai))

    def show_exercise_selection(self, current_values):
        """Показать экран выбора упражнения"""
        self.layout.clear_widgets()
        self.layout.add_widget(TrainingScreen(current_values, self.exercise_ai))

    def show_profile_screen(self):
        """Показать личный кабинет"""
        self.layout.clear_widgets()
        self.layout.add_widget(ProfileScreen(self.profile, self.show_progress_screen))

    def start_exercise_session(self, exercise, target):
        """Запуск сессии упражнения"""
        self.layout.clear_widgets()
        self.layout.add_widget(ExerciseSessionScreen(
            self.exercise_ai,
            exercise,
            target
        ))

    def update_exercise_value(self, exercise, increment):
        """Обновление значения упражнения"""
        if exercise in self.current_values:
            self.current_values[exercise] += increment
            print(f"✓ {exercise} увеличен на {increment}, теперь {self.current_values[exercise]}")

    def add_workout_to_profile(self, exercise, reps):
        """Добавление тренировки в профиль"""
        self.profile.add_workout(exercise, reps)

    def show_error_popup(self, message):
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        content.add_widget(Label(text=message, font_size='16sp'))

        btn = Button(text='OK', size_hint=(1, 0.3))
        popup = Popup(title='Ошибка', content=content, size_hint=(0.6, 0.3))
        btn.bind(on_press=popup.dismiss)
        content.add_widget(btn)
        popup.open()

    def show_warning_popup(self, message):
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        content.add_widget(Label(
            text=message,
            font_size='16sp',
            color=(1, 1, 0, 1)
        ))

        btn = Button(text='Понятно', size_hint=(1, 0.3))
        popup = Popup(title='Предупреждение', content=content, size_hint=(0.7, 0.4))
        btn.bind(on_press=popup.dismiss)
        content.add_widget(btn)
        popup.open()

    def show_success_popup(self, message):
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        content.add_widget(Label(
            text=message,
            font_size='16sp',
            color=(0, 1, 0, 1)
        ))

        btn = Button(text='Отлично!', size_hint=(1, 0.3))
        popup = Popup(title='Успех!', content=content, size_hint=(0.6, 0.3))
        btn.bind(on_press=popup.dismiss)
        content.add_widget(btn)
        popup.open()

    # Добавьте в класс Mobile_Trainer:
    def show_convert_menu(self):
        """Меню конвертации моделей"""
        content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))

        content.add_widget(Label(
            text="Выберите модель для конвертации:",
            font_size='16sp'
        ))

        btn_layout = BoxLayout(orientation='vertical', spacing=dp(5), size_hint_y=None)
        btn_layout.height = dp(200)

        models = [
            ("Отжимания", self.exercise_ai.PUSHUP_MODEL_PATH),
            ("Подтягивания", self.exercise_ai.PULLUP_MODEL_PATH),
            ("Пресс", self.exercise_ai.PRESS_MODEL_PATH),
            ("Приседания", self.exercise_ai.SQUAT_MODEL_PATH)
        ]

        for name, path in models:
            if os.path.exists(path):
                btn = Button(
                    text=f"Конвертировать {name}",
                    size_hint_y=None,
                    height=dp(40)
                )
                btn.bind(on_press=lambda x, p=path: self.convert_and_reload(p))
                btn_layout.add_widget(btn)

        content.add_widget(btn_layout)

        close_btn = Button(text="Закрыть", size_hint_y=None, height=dp(40))
        popup = Popup(title="Конвертация моделей", content=content, size_hint=(0.6, 0.5))
        close_btn.bind(on_press=popup.dismiss)
        content.add_widget(close_btn)

        popup.open()

    def convert_and_reload(self, model_path):
        """Конвертирует модель и перезагружает её"""
        success = self.exercise_ai.convert_model_for_compatibility(model_path)
        if success:
            self.show_success_popup("Модель сконвертирована успешно!\nПерезапустите приложение.")
        else:
            self.show_error_popup("Ошибка конвертации модели")


if __name__ == "__main__":
    try:
        Mobile_Trainer().run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        input("Нажмите Enter для выхода...")