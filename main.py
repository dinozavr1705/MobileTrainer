#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Решение проблемы OpenGL
if sys.platform == 'darwin':
    os.environ['KIVY_GL_BACKEND'] = 'sdl2'
else:
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

import cv2
import numpy as np
import datetime
import json
import warnings
from pathlib import Path
import threading
import time

# Определяем корневую директорию приложения
def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = get_app_dir()
MODELS_DIR = os.path.join(APP_DIR, "saved_models")
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Пути к файлам профиля
PROFILE_FILE = os.path.join(DATA_DIR, "user_profile.json")
LAST_MAX_UPDATE_FILE = os.path.join(DATA_DIR, "last_max_update.json")

# Пути к моделям
def get_model_path(filename):
    return os.path.join(APP_DIR, filename)

def get_scaler_path(model_path):
    return model_path.replace('.keras', '_scaler.pkl')

# Определяем эмулятор
IN_ANDROID_EMULATOR = False
if os.path.exists('/system/build.prop') or 'ANDROID_ROOT' in os.environ:
    IN_ANDROID_EMULATOR = True
    print("⚠ Обнаружен Android-эмулятор")

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    import joblib
    TF_AVAILABLE = True
    print(f"✓ TensorFlow {tf.__version__} загружен")
except ImportError as e:
    TF_AVAILABLE = False
    print(f"✗ TensorFlow не загружен: {e}")

warnings.filterwarnings('ignore')
if TF_AVAILABLE:
    tf.get_logger().setLevel('ERROR')


class UserProfile:
    def __init__(self):
        self.profile_file = PROFILE_FILE
        self.last_max_update_file = LAST_MAX_UPDATE_FILE
        self.start_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.total_days = 1
        self.total_workouts = 0
        self.total_reps = {"отжимания": 0, "подтягивания": 0, "пресс": 0, "приседания": 0}
        self.max_reps = {"отжимания": 0, "подтягивания": 0, "пресс": 0, "приседания": 0}
        self.original_max = {"отжимания": 0, "подтягивания": 0, "пресс": 0, "приседания": 0}
        self.current_targets = {"отжимания": 0, "подтягивания": 0, "пресс": 0, "приседания": 0}
        self.workout_history = []
        self.last_workout_date = None
        self.streak_days = 0
        self.last_max_update = None
        self.load_profile()
        self.load_last_max_update()

    def load_profile(self):
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
                print("✓ Профиль загружен")
            except Exception as e:
                print(f"Ошибка загрузки профиля: {e}")

    def save_profile(self):
        try:
            data = {
                'start_date': self.start_date,
                'total_days': self.total_days,
                'total_workouts': self.total_workouts,
                'total_reps': self.total_reps,
                'max_reps': self.max_reps,
                'original_max': self.original_max,
                'current_targets': self.current_targets,
                'workout_history': self.workout_history[-100:],
                'last_workout_date': self.last_workout_date,
                'streak_days': self.streak_days
            }
            with open(self.profile_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("✓ Профиль сохранен")
        except Exception as e:
            print(f"Ошибка сохранения профиля: {e}")

    def load_last_max_update(self):
        if os.path.exists(self.last_max_update_file):
            try:
                with open(self.last_max_update_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.last_max_update = data.get('last_update')
                print("✓ Дата обновления загружена")
            except Exception as e:
                print(f"Ошибка: {e}")

    def save_last_max_update(self):
        try:
            data = {'last_update': datetime.datetime.now().strftime("%Y-%m-%d")}
            with open(self.last_max_update_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.last_max_update = data['last_update']
            print("✓ Дата обновления сохранена")
        except Exception as e:
            print(f"Ошибка: {e}")

    def needs_max_update(self):
        if self.last_max_update is None:
            return True
        try:
            last = datetime.datetime.strptime(self.last_max_update, "%Y-%m-%d")
            days_passed = (datetime.datetime.now() - last).days
            return days_passed >= 7
        except:
            return True

    def update_max_values(self, new_max_values):
        self.original_max = new_max_values.copy()
        self.max_reps = new_max_values.copy()
        for ex in new_max_values:
            self.current_targets[ex] = new_max_values[ex]
        self.save_last_max_update()
        self.save_profile()
        print("✓ Максимумы обновлены")

    def increase_target(self, exercise):
        if exercise in self.current_targets and exercise in self.original_max:
            increment = int(self.original_max[exercise] * 2.5)
            self.current_targets[exercise] += increment
            self.save_profile()
            return True
        return False

    def add_workout(self, exercise, reps):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.total_workouts += 1
        self.total_reps[exercise] += reps
        if reps > self.max_reps[exercise]:
            self.max_reps[exercise] = reps
        self.workout_history.append({'date': today, 'exercise': exercise, 'reps': reps})
        if self.last_workout_date == today:
            pass
        elif self.last_workout_date == self.get_yesterday():
            self.streak_days += 1
        else:
            self.streak_days = 1
        self.last_workout_date = today
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        self.total_days = (datetime.datetime.now() - start).days + 1
        self.save_profile()

    def get_yesterday(self):
        return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    def get_week_stats(self):
        week_stats = {ex: 0 for ex in self.total_reps.keys()}
        today = datetime.datetime.now()
        for i in range(7):
            day = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            for workout in self.workout_history:
                if workout['date'] == day:
                    week_stats[workout['exercise']] += workout['reps']
        return week_stats


class CameraStream:
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

    def start(self):
        if not self.stopped:
            return self
        if IN_ANDROID_EMULATOR:
            self.use_fake_camera = True
            self.stopped = False
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._update_fake, daemon=True)
            self.thread.start()
            return self
        try:
            if platform == 'win':
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            elif platform == 'darwin':
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(self.src)
            if not self.cap or not self.cap.isOpened():
                self.use_fake_camera = True
                self.stopped = False
                self.thread = threading.Thread(target=self._update_fake, daemon=True)
                self.thread.start()
                return self
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.stopped = False
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            print("✓ Камера запущена")
        except Exception as e:
            print(f"✗ Ошибка: {e}")
            self.use_fake_camera = True
            self.stopped = False
            self.thread = threading.Thread(target=self._update_fake, daemon=True)
            self.thread.start()
        return self

    def _update(self):
        while not self.stopped and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
                    self.frame_count += 1
                else:
                    time.sleep(0.01)
            except:
                time.sleep(0.01)

    def _update_fake(self):
        while not self.stopped:
            try:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.fake_frame_index += 1
                phase = self.fake_frame_index / 20.0
                head_y = 150 + int(20 * np.sin(phase))
                cv2.circle(frame, (320, head_y), 30, (100, 150, 200), -1)
                cv2.circle(frame, (300, head_y-10), 5, (255,255,255), -1)
                cv2.circle(frame, (340, head_y-10), 5, (255,255,255), -1)
                body_y = 200 + int(30 * np.sin(phase))
                cv2.line(frame, (320, head_y+20), (320, body_y), (150,150,150), 20)
                arm_angle = np.sin(phase*2)*0.8
                left_arm = 320 - int(60 * abs(np.cos(arm_angle)))
                right_arm = 320 + int(60 * abs(np.cos(arm_angle)))
                cv2.line(frame, (320, head_y+30), (left_arm, body_y-20), (150,150,150), 10)
                cv2.line(frame, (320, head_y+30), (right_arm, body_y-20), (150,150,150), 10)
                cv2.putText(frame, "ДЕМО РЕЖИМ", (200,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                position = "ВНИЗУ" if abs(np.sin(phase)) > 0.7 else "ВВЕРХУ"
                color = (0,0,255) if position == "ВНИЗУ" else (0,255,0)
                cv2.putText(frame, f"Положение: {position}", (200,440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                self.ret = True
                self.frame = frame
                self.frame_count += 1
                time.sleep(1.0/30.0)
            except:
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


class ExerciseClassifier:
    def __init__(self):
        self.classes = ['отжимания', 'подтягивания', 'пресс', 'приседания']

    def classify(self, frame):
        try:
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            ar = h / w
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_ratio = np.mean(np.abs(sobelx)) / (np.mean(np.abs(sobely)) + 1e-6)
            top = np.mean(gray[:h//3, :])
            middle = np.mean(gray[h//3:2*h//3, :])
            bottom = np.mean(gray[2*h//3:, :])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skin = cv2.inRange(hsv, np.array([0,20,70]), np.array([20,255,255]))
            skin_pct = np.mean(skin) / 255
            scores = {}
            scores['отжимания'] = (0.3 if ar < 1.2 else 0) + (0.3 if grad_ratio > 1.2 else 0) + (0.2 if skin_pct > 0.3 else 0) + (0.2 if middle < top and middle < bottom else 0)
            scores['подтягивания'] = (0.3 if ar > 1.6 else 0) + (0.3 if grad_ratio < 0.8 else 0) + (0.2 if skin_pct < 0.2 else 0) + (0.2 if top < bottom else 0)
            scores['пресс'] = (0.2 if ar < 1.3 else 0)
            edges = cv2.Canny(gray, 50, 150)
            top_edges = np.mean(edges[:h//2, :]) / 255
            scores['пресс'] += (0.4 if top_edges > 0.1 else 0) + (0.2 if top > bottom else 0)
            scores['приседания'] = (0.3 if 1.2 < ar < 1.6 else 0) + (0.4 if middle < top and middle < bottom else 0) + (0.3 if skin_pct > 0.25 else 0)
            total = sum(scores.values())
            if total > 0:
                for k in scores:
                    scores[k] /= total
            best = max(scores, key=scores.get)
            return best, scores[best], scores
        except:
            return "неизвестно", 0, {}


class ExerciseAI:
    def __init__(self):
        self.app_dir = APP_DIR
        self.models_dir = MODELS_DIR
        self.PUSHUP_MODEL_PATH = get_model_path("push_up_model.keras")
        self.PULLUP_MODEL_PATH = get_model_path("pull_up_model.keras")
        self.PRESS_MODEL_PATH = get_model_path("press_model.keras")
        self.SQUAT_MODEL_PATH = get_model_path("squat_model.keras")
        self.pushup_scaler_path = get_scaler_path(self.PUSHUP_MODEL_PATH)
        self.pullup_scaler_path = get_scaler_path(self.PULLUP_MODEL_PATH)
        self.press_scaler_path = get_scaler_path(self.PRESS_MODEL_PATH)
        self.squat_scaler_path = get_scaler_path(self.SQUAT_MODEL_PATH)
        print(f"Отжимания модель: {os.path.exists(self.PUSHUP_MODEL_PATH)}")
        print(f"Подтягивания модель: {os.path.exists(self.PULLUP_MODEL_PATH)}")
        print(f"Пресс модель: {os.path.exists(self.PRESS_MODEL_PATH)}")
        print(f"Приседания модель: {os.path.exists(self.SQUAT_MODEL_PATH)}")
        self.pushup_model = None
        self.pullup_model = None
        self.press_model = None
        self.squat_model = None
        self.pushup_scaler = None
        self.pullup_scaler = None
        self.press_scaler = None
        self.squat_scaler = None
        self.pushup_count = 0
        self.pullup_count = 0
        self.press_count = 0
        self.squat_count = 0
        self.pushup_state = "up"
        self.pullup_state = "down"
        self.press_state = "down"
        self.squat_state = "up"
        self.demo_mode = not TF_AVAILABLE or IN_ANDROID_EMULATOR
        self.models_loaded = False
        self.classifier = ExerciseClassifier()
        self.model_statuses = {"отжимания": "⏳ Загрузка...", "подтягивания": "⏳ Загрузка...", "пресс": "⏳ Загрузка...", "приседания": "⏳ Загрузка..."}
        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        print("\n" + "="*60)
        print("ЗАГРУЗКА МОДЕЛЕЙ ИИ")
        print("="*60)
        if not TF_AVAILABLE:
            self.demo_mode = True
            for ex in self.model_statuses:
                self.model_statuses[ex] = "❌ TensorFlow не установлен"
            self.models_loaded = True
            return
        if os.path.exists(self.PUSHUP_MODEL_PATH):
            try:
                self.pushup_model = keras.models.load_model(self.PUSHUP_MODEL_PATH, compile=False)
                if os.path.exists(self.pushup_scaler_path):
                    self.pushup_scaler = joblib.load(self.pushup_scaler_path)
                    self.model_statuses["отжимания"] = "✅ ИИ активен"
                else:
                    self.model_statuses["отжимания"] = "⚠ Нет scaler"
                    self.pushup_model = None
            except Exception as e:
                print(f"✗ Ошибка: {e}")
                self.model_statuses["отжимания"] = "❌ Ошибка"
        else:
            self.model_statuses["отжимания"] = "❌ Файл не найден"
        if os.path.exists(self.PULLUP_MODEL_PATH):
            try:
                self.pullup_model = keras.models.load_model(self.PULLUP_MODEL_PATH, compile=False)
                if os.path.exists(self.pullup_scaler_path):
                    self.pullup_scaler = joblib.load(self.pullup_scaler_path)
                    self.model_statuses["подтягивания"] = "✅ ИИ активен"
                else:
                    self.model_statuses["подтягивания"] = "⚠ Нет scaler"
                    self.pullup_model = None
            except:
                self.model_statuses["подтягивания"] = "❌ Ошибка"
        else:
            self.model_statuses["подтягивания"] = "❌ Файл не найден"
        if os.path.exists(self.PRESS_MODEL_PATH):
            try:
                self.press_model = keras.models.load_model(self.PRESS_MODEL_PATH, compile=False)
                if os.path.exists(self.press_scaler_path):
                    self.press_scaler = joblib.load(self.press_scaler_path)
                    self.model_statuses["пресс"] = "✅ ИИ активен"
                else:
                    self.model_statuses["пресс"] = "⚠ Нет scaler"
                    self.press_model = None
            except:
                self.model_statuses["пресс"] = "❌ Ошибка"
        else:
            self.model_statuses["пресс"] = "❌ Файл не найден"
        if os.path.exists(self.SQUAT_MODEL_PATH):
            try:
                self.squat_model = keras.models.load_model(self.SQUAT_MODEL_PATH, compile=False)
                if os.path.exists(self.squat_scaler_path):
                    self.squat_scaler = joblib.load(self.squat_scaler_path)
                    self.model_statuses["приседания"] = "✅ ИИ активен"
                else:
                    self.model_statuses["приседания"] = "⚠ Нет scaler"
                    self.squat_model = None
            except:
                self.model_statuses["приседания"] = "❌ Ошибка"
        else:
            self.model_statuses["приседания"] = "❌ Файл не найден"
        self.models_loaded = True
        self.demo_mode = not (self.pushup_model or self.pullup_model or self.press_model or self.squat_model)
        print("\n" + "="*60)
        print("СТАТУС МОДЕЛЕЙ:")
        print(f"Отжимания: {'✅' if self.pushup_model else '❌'}")
        print(f"Подтягивания: {'✅' if self.pullup_model else '❌'}")
        print(f"Пресс: {'✅' if self.press_model else '❌'}")
        print(f"Приседания: {'✅' if self.squat_model else '❌'}")
        print(f"Режим: {'ДЕМО' if self.demo_mode else 'ИИ'}")
        print("="*60)

    def check_model_available(self, exercise):
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
        try:
            frame = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            f = []
            f.extend([np.mean(gray), np.std(gray), np.median(gray)])
            skin = cv2.inRange(hsv, np.array([0,20,70]), np.array([20,255,255]))
            f.append(np.mean(skin)/255)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            f.extend([np.mean(np.abs(sx)), np.mean(np.abs(sy)), np.std(sx), np.std(sy)])
            mag, ang = cv2.cartToPolar(sx, sy)
            hst, _ = np.histogram(ang, bins=4, range=(0, 2*np.pi))
            hst = hst / (hst.sum() + 1e-6)
            f.extend(hst)
            edges = cv2.Canny(gray, 50, 150)
            f.append(np.mean(edges)/255)
            cnt, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnt:
                c = max(cnt, key=cv2.contourArea)
                f.append(cv2.contourArea(c) / (cv2.arcLength(c, True) + 1e-6))
            else:
                f.append(0)
            while len(f) < 20:
                f.append(0)
            return np.array(f[:20])
        except:
            return np.zeros(20)

    def extract_pullup_features(self, frame):
        try:
            frame = cv2.resize(frame, (320, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = gray.shape
            f = [h/w]
            tz = gray[:h//3, :]
            mz = gray[h//3:2*h//3, :]
            bz = gray[2*h//3:, :]
            f.extend([np.mean(tz), np.mean(mz), np.mean(bz), np.mean(tz)-np.mean(bz)])
            edges = cv2.Canny(gray, 50, 150)
            f.append(np.mean(edges[:h//4, :])/255)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            f.extend([np.mean(sy), np.std(sy)])
            skin = cv2.inRange(hsv, np.array([0,20,70]), np.array([25,255,255]))
            f.append(np.mean(skin)/255)
            lh = gray[:, :w//2]
            rh = gray[:, w//2:]
            f.append(np.mean(np.abs(lh - rh)) if lh.shape == rh.shape else 0)
            _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            f.extend([np.mean(th[:h//3, :])/255, np.mean(th[h//3:2*h//3, :])/255, np.mean(th[2*h//3:, :])/255])
            f.extend([np.max(gray)-np.min(gray)])
            hist = cv2.calcHist([gray], [0], None, [16], [0,256])
            hn = hist / (hist.sum() + 1e-6)
            f.append(-np.sum(hn * np.log2(hn + 1e-10)))
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            f.append(np.mean(np.abs(sy)) / (np.mean(np.abs(sx)) + 1e-6))
            while len(f) < 18:
                f.append(0)
            return np.array(f[:18])
        except:
            return np.zeros(18)

    def extract_press_features(self, frame):
        try:
            frame = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            f = [np.mean(gray), np.std(gray), np.median(gray)]
            skin = cv2.inRange(hsv, np.array([0,20,70]), np.array([20,255,255]))
            f.append(np.mean(skin)/255)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            f.extend([np.mean(np.abs(sx)), np.mean(np.abs(sy)), np.std(sx), np.std(sy)])
            mag, ang = cv2.cartToPolar(sx, sy)
            hst, _ = np.histogram(ang, bins=4, range=(0, 2*np.pi))
            hst = hst / (hst.sum() + 1e-6)
            f.extend(hst)
            edges = cv2.Canny(gray, 50, 150)
            f.append(np.mean(edges)/255)
            h, w = gray.shape
            f.extend([np.mean(edges[:h//2, :])/255, np.mean(edges[h//2:, :])/255])
            while len(f) < 20:
                f.append(0)
            return np.array(f[:20])
        except:
            return np.zeros(20)

    def extract_squat_features(self, frame):
        try:
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = gray.shape
            f = [h/w, np.mean(gray), np.std(gray)]
            f.extend([np.mean(gray[:, :w//3]), np.mean(gray[:, w//3:2*w//3]), np.mean(gray[:, 2*w//3:])])
            f.extend([np.mean(gray[:h//3, :]), np.mean(gray[h//3:2*h//3, :]), np.mean(gray[2*h//3:, :])])
            f.extend([np.mean(gray[:h//3, :])/(np.mean(gray[2*h//3:, :])+1e-6)])
            f.extend([np.mean(gray[:, w//3:2*w//3])/(np.mean(gray[:, :w//3])+1e-6)])
            f.extend([np.mean(gray[:, w//3:2*w//3])/(np.mean(gray[:, 2*w//3:])+1e-6)])
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            f.extend([np.mean(np.abs(sx)), np.std(np.abs(sx)), np.mean(np.abs(sy)), np.std(np.abs(sy))])
            f.append(np.mean(np.abs(sx)) / (np.mean(np.abs(sy)) + 1e-6))
            while len(f) < 20:
                f.append(0)
            return np.array(f[:20])
        except:
            return np.zeros(20)

    def analyze_pushup_frame(self, frame):
        if self.check_model_available("отжимания"):
            try:
                f = self.extract_pushup_features(frame)
                fa = np.array([f])
                cmb = np.concatenate([np.mean(fa,0), np.std(fa,0), np.max(fa,0), np.min(fa,0)])
                if len(cmb) != 80:
                    cmb = np.pad(cmb, (0, max(0, 80-len(cmb))))[:80]
                if self.pushup_scaler:
                    fs = self.pushup_scaler.transform([cmb])
                else:
                    fs = [cmb]
                p = self.pushup_model.predict(fs, verbose=0)[0][0]
                return {'prediction': 'down' if p>0.5 else 'up', 'confidence': p if p>0.5 else 1-p, 'probability': p, 'demo': False}
            except:
                pass
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p = 1.0 - (np.mean(g)/255.0)
        p = np.clip(p, 0.2, 0.9)
        return {'prediction': 'down' if p>0.6 else 'up', 'confidence': abs(p-0.5)*2, 'probability': p, 'demo': True}

    def analyze_pullup_frame(self, frame):
        if self.check_model_available("подтягивания"):
            try:
                f = self.extract_pullup_features(frame)
                fa = np.array([f])
                cmb = np.concatenate([np.mean(fa,0), np.std(fa,0), np.max(fa,0), np.min(fa,0)])
                if len(cmb) != 72:
                    cmb = np.pad(cmb, (0, max(0, 72-len(cmb))))[:72]
                if self.pullup_scaler:
                    fs = self.pullup_scaler.transform([cmb])
                else:
                    fs = [cmb]
                p = self.pullup_model.predict(fs, verbose=0)[0][0]
                return {'prediction': 'up' if p>0.5 else 'down', 'confidence': p if p>0.5 else 1-p, 'probability': p, 'demo': False}
            except:
                pass
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 50, 150)
        p = np.clip(np.sum(e) / (frame.shape[0]*frame.shape[1]+1) * 3, 0.2, 0.9)
        return {'prediction': 'up' if p>0.6 else 'down', 'confidence': abs(p-0.5)*2, 'probability': p, 'demo': True}

    def analyze_press_frame(self, frame):
        if self.check_model_available("пресс"):
            try:
                f = self.extract_press_features(frame)
                fa = np.array([f])
                cmb = np.concatenate([np.mean(fa,0), np.std(fa,0), np.max(fa,0), np.min(fa,0)])
                if len(cmb) != 80:
                    cmb = np.pad(cmb, (0, max(0, 80-len(cmb))))[:80]
                if self.press_scaler:
                    fs = self.press_scaler.transform([cmb])
                else:
                    fs = [cmb]
                p = self.press_model.predict(fs, verbose=0)[0][0]
                return {'prediction': 'up' if p>0.5 else 'down', 'confidence': p if p>0.5 else 1-p, 'probability': p, 'demo': False}
            except:
                pass
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = g.shape[0]
        e = cv2.Canny(g, 50, 150)
        p = np.clip(np.sum(e[h//2:, :]) / (e[h//2:, :].size + 1) * 4, 0.2, 0.9)
        return {'prediction': 'up' if p>0.6 else 'down', 'confidence': abs(p-0.5)*2, 'probability': p, 'demo': True}

    def analyze_squat_frame(self, frame):
        if self.check_model_available("приседания"):
            try:
                f = self.extract_squat_features(frame)
                fa = np.array([f])
                cmb = np.concatenate([np.mean(fa,0), np.std(fa,0), np.max(fa,0), np.min(fa,0)])
                if len(cmb) != 80:
                    cmb = np.pad(cmb, (0, max(0, 80-len(cmb))))[:80]
                if self.squat_scaler:
                    fs = self.squat_scaler.transform([cmb])
                else:
                    fs = [cmb]
                p = self.squat_model.predict(fs, verbose=0)[0][0]
                return {'prediction': 'down' if p>0.5 else 'up', 'confidence': p if p>0.5 else 1-p, 'probability': p, 'demo': False}
            except:
                pass
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = g.shape[0]
        e = cv2.Canny(g, 50, 150)
        p = np.clip(np.sum(e[h//2:, :]) / (e[h//2:, :].size + 1) * 3, 0.2, 0.9)
        return {'prediction': 'down' if p>0.6 else 'up', 'confidence': abs(p-0.5)*2, 'probability': p, 'demo': True}

    def detect_pushup_rep(self, pred):
        if not pred:
            return False
        if pred['prediction'] == 'down' and self.pushup_state == "up":
            self.pushup_state = "down"
            return False
        elif pred['prediction'] == 'up' and self.pushup_state == "down":
            self.pushup_state = "up"
            self.pushup_count += 1
            return True
        return False

    def detect_pullup_rep(self, pred):
        if not pred:
            return False
        if pred['prediction'] == 'up' and self.pullup_state == "down":
            self.pullup_state = "up"
            return False
        elif pred['prediction'] == 'down' and self.pullup_state == "up":
            self.pullup_state = "down"
            self.pullup_count += 1
            return True
        return False

    def detect_press_rep(self, pred):
        if not pred:
            return False
        if pred['prediction'] == 'up' and self.press_state == "down":
            self.press_state = "up"
            return False
        elif pred['prediction'] == 'down' and self.press_state == "up":
            self.press_state = "down"
            self.press_count += 1
            return True
        return False

    def detect_squat_rep(self, pred):
        if not pred:
            return False
        if pred['prediction'] == 'down' and self.squat_state == "up":
            self.squat_state = "down"
            return False
        elif pred['prediction'] == 'up' and self.squat_state == "down":
            self.squat_state = "up"
            self.squat_count += 1
            return True
        return False

    def reset_counts(self):
        self.pushup_count = 0
        self.pullup_count = 0
        self.press_count = 0
        self.squat_count = 0
        self.pushup_state = "up"
        self.pullup_state = "down"
        self.press_state = "down"
        self.squat_state = "up"

    def get_count(self, ex):
        return {"отжимания": self.pushup_count, "подтягивания": self.pullup_count, "пресс": self.press_count, "приседания": self.squat_count}.get(ex, 0)

    def get_model_status(self, ex):
        return self.model_statuses.get(ex, "ℹ️ Демо-режим") if self.models_loaded else "⏳ Загрузка..."

    def verify_exercise(self, frame, expected):
        det, conf, _ = self.classifier.classify(frame)
        return (det == expected, conf, f"✓ Обнаружено {det}" if det == expected else f"⚠ Обнаружено {det}, а выбрано {expected}")


class KivyCamera(Image):
    def __init__(self, exercise_ai, exercise_name, on_rep_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.exercise_ai = exercise_ai
        self.exercise_name = exercise_name
        self.on_rep_callback = on_rep_callback
        self.camera_stream = None
        self.current_count = 0
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
            self.update_event = Clock.schedule_interval(self.update, 1.0/30.0)
        except Exception as e:
            self.camera_available = False
            print(f"✗ Ошибка камеры: {e}")

    def update(self, dt):
        if not self.camera_available or not self.camera_stream:
            return
        try:
            ret, frame = self.camera_stream.read()
            if not ret or frame is None:
                return
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            self.verification_frames += 1
            if self.verification_frames % 30 == 0:
                is_correct, _, msg = self.exercise_ai.verify_exercise(frame, self.exercise_name)
                if not is_correct and not self.misalignment_warning_shown:
                    self.misalignment_warning_shown = True
                    App.get_running_app().show_warning_popup(msg)
            pred = None
            is_rep = False
            if self.exercise_name == "отжимания":
                pred = self.exercise_ai.analyze_pushup_frame(frame)
                is_rep = self.exercise_ai.detect_pushup_rep(pred)
                self.current_count = self.exercise_ai.pushup_count
            elif self.exercise_name == "подтягивания":
                pred = self.exercise_ai.analyze_pullup_frame(frame)
                is_rep = self.exercise_ai.detect_pullup_rep(pred)
                self.current_count = self.exercise_ai.pullup_count
            elif self.exercise_name == "пресс":
                pred = self.exercise_ai.analyze_press_frame(frame)
                is_rep = self.exercise_ai.detect_press_rep(pred)
                self.current_count = self.exercise_ai.press_count
            elif self.exercise_name == "приседания":
                pred = self.exercise_ai.analyze_squat_frame(frame)
                is_rep = self.exercise_ai.detect_squat_rep(pred)
                self.current_count = self.exercise_ai.squat_count
            if is_rep:
                self.rep_animation = 10
                self.misalignment_warning_shown = False
                if self.on_rep_callback:
                    self.on_rep_callback()
            status = self.exercise_ai.get_model_status(self.exercise_name)
            cv2.putText(frame, f"Status: {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"СЧЕТ: {self.current_count}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
            if pred:
                color = (0,255,0) if pred['prediction'] == 'up' else (0,0,255)
                txt = "DEMO" if pred.get('demo', False) else "AI"
                pos = "ВВЕРХУ" if pred['prediction'] == 'up' else "ВНИЗУ"
                cv2.putText(frame, f"{txt}: {pos}", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if self.rep_animation > 0:
                cv2.putText(frame, "ПОВТОРЕНИЕ!", (200,300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 4)
                self.rep_animation -= 1
            if self.misalignment_warning_shown:
                cv2.putText(frame, "⚠ НЕПРАВИЛЬНОЕ УПРАЖНЕНИЕ!", (150,400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            buf = cv2.flip(frame, 0).tobytes()
            tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = tex
        except Exception as e:
            print(f"Ошибка: {e}")

    def stop_camera(self):
        if self.update_event:
            self.update_event.cancel()
        if self.camera_stream:
            self.camera_stream.stop()
        self.camera_stream = None


class ProfileScreen(BoxLayout):
    def __init__(self, profile, on_back, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.padding = dp(20)
        self.profile = profile
        self.on_back = on_back
        self.add_widget(Label(text="👤 ЛИЧНЫЙ КАБИНЕТ", font_size='28sp', bold=True, color=(0.2,0.8,1,1), size_hint=(1,0.1), halign='center'))
        stats = BoxLayout(orientation='vertical', size_hint=(1,0.25), spacing=dp(10))
        card = BoxLayout(orientation='vertical', size_hint=(1,1), padding=dp(15))
        with card.canvas.before:
            Color(0.15,0.15,0.15,1)
            card.rect = Rectangle(size=card.size, pos=card.pos)
        card.bind(pos=self._update_rect, size=self._update_rect)
        total_txt = (f"[color=FFD700]📅 Дней:[/color] [b]{profile.total_days}[/b]\n"
                     f"[color=FFD700]🔥 Серия:[/color] [b]{profile.streak_days}[/b] дн\n"
                     f"[color=FFD700]💪 Тренировок:[/color] [b]{profile.total_workouts}[/b]\n"
                     f"[color=FFD700]🎯 Повторений:[/color] [b]{sum(profile.total_reps.values())}[/b]")
        card.add_widget(Label(text=total_txt, font_size='16sp', markup=True, halign='left', valign='middle', color=(1,1,1,1)))
        stats.add_widget(card)
        self.add_widget(stats)
        self.add_widget(Label(text="🎯 ТЕКУЩИЕ ЦЕЛИ", font_size='18sp', bold=True, color=(0.2,0.8,1,1), size_hint=(1,0.05), halign='center'))
        tg = GridLayout(cols=2, spacing=dp(15), padding=dp(10), size_hint=(1,0.25))
        for name, key, col in [("💪 Отжимания","отжимания",(0.3,0.6,0.3,0.8)),("⬆️ Подтягивания","подтягивания",(0.3,0.3,0.6,0.8)),("🔄 Пресс","пресс",(0.6,0.3,0.3,0.8)),("🦵 Приседания","приседания",(0.6,0.6,0.3,0.8))]:
            c = BoxLayout(orientation='vertical', padding=dp(10))
            with c.canvas.before:
                Color(*col)
                c.rect = Rectangle(size=c.size, pos=c.pos)
            c.bind(pos=self._update_rect, size=self._update_rect)
            cur = profile.current_targets.get(key,0)
            orig = profile.original_max.get(key,0)
            prog = int((cur/max(1,orig))*100) if orig>0 else 0
            txt = (f"[b]{name}[/b]\n\n[color=FFFFFF]🎯 Цель:[/color] [b]{cur}[/b]\n[color=AAAAAA]📊 Исх:[/color] {orig}\n[color=00FF00]📈 Прогресс:[/color] {prog}%")
            c.add_widget(Label(text=txt, markup=True, font_size='14sp', halign='center', valign='middle', color=(1,1,1,1)))
            tg.add_widget(c)
        self.add_widget(tg)
        self.add_widget(Label(text="📊 СТАТИСТИКА ПО УПРАЖНЕНИЯМ", font_size='18sp', bold=True, color=(0.2,0.8,1,1), size_hint=(1,0.05), halign='center'))
        gr = GridLayout(cols=2, spacing=dp(15), padding=dp(10), size_hint=(1,0.25))
        for name, key, col in [("💪 Отжимания","отжимания",(0.3,0.6,0.3,0.8)),("⬆️ Подтягивания","подтягивания",(0.3,0.3,0.6,0.8)),("🔄 Пресс","пресс",(0.6,0.3,0.3,0.8)),("🦵 Приседания","приседания",(0.6,0.6,0.3,0.8))]:
            c = BoxLayout(orientation='vertical', padding=dp(10))
            with c.canvas.before:
                Color(*col)
                c.rect = Rectangle(size=c.size, pos=c.pos)
            c.bind(pos=self._update_rect, size=self._update_rect)
            tot = profile.total_reps.get(key,0)
            mx = profile.max_reps.get(key,0)
            txt = f"[b]{name}[/b]\n\n[color=FFFFFF]📊 Всего:[/color] [b]{tot}[/b]\n[color=FFD700]🏆 Макс:[/color] [b]{mx}[/b]"
            c.add_widget(Label(text=txt, markup=True, font_size='14sp', halign='center', valign='middle', color=(1,1,1,1)))
            gr.add_widget(c)
        self.add_widget(gr)
        back = Button(text="◀️ НАЗАД К ПРОГРЕССУ", font_size='18sp', size_hint=(1,0.1), background_color=(0.5,0.5,0.5,1), color=(1,1,1,1))
        back.bind(on_press=self.go_back)
        self.add_widget(back)

    def _update_rect(self, ins, val):
        ins.canvas.before.clear()
        with ins.canvas.before:
            Color(0.15,0.15,0.15,1)
            Rectangle(size=ins.size, pos=ins.pos)

    def go_back(self, ins):
        self.on_back()


class MaxInputScreen(BoxLayout):
    def __init__(self, profile, on_submit, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(20)
        self.padding = dp(20)
        self.profile = profile
        self.on_submit = on_submit
        if not profile.needs_max_update():
            Clock.schedule_once(lambda dt: self.on_submit(profile.original_max), 0)
            return
        self.add_widget(Label(text="🏋️ ОБНОВИТЕ ВАШ МАКСИМУМ", font_size='24sp', bold=True, color=(0.2,0.8,1,1), size_hint=(1,0.15)))
        self.add_widget(Label(text="Прошла неделя! Укажите новое максимальное количество повторений", font_size='16sp', halign='center', size_hint=(1,0.1)))
        self.inputs = {}
        for key, label, default in [("отжимания","💪 Отжимания",profile.original_max.get("отжимания",50)),("подтягивания","⬆️ Подтягивания",profile.original_max.get("подтягивания",10)),("пресс","🔄 Пресс",profile.original_max.get("пресс",30)),("приседания","🦵 Приседания",profile.original_max.get("приседания",50))]:
            box = BoxLayout(orientation='horizontal', size_hint=(1,0.12), spacing=dp(10))
            box.add_widget(Label(text=label, font_size='16sp', size_hint=(0.4,1), halign='left'))
            inp = TextInput(text=str(default), font_size='18sp', multiline=False, input_filter='int', size_hint=(0.3,1))
            box.add_widget(inp)
            self.inputs[key] = inp
            self.add_widget(box)
        btn = Button(text="✅ СОХРАНИТЬ МАКСИМУМ", font_size='20sp', background_color=(0.2,0.8,0.2,1), size_hint=(1,0.15))
        btn.bind(on_press=self.submit)
        self.add_widget(btn)

    def submit(self, ins):
        try:
            vals = {k: int(v.text.strip()) if v.text.strip() else 0 for k,v in self.inputs.items()}
            self.profile.update_max_values(vals)
            self.on_submit(vals)
        except:
            App.get_running_app().show_error_popup("Введите корректные числа!")


class ExerciseCard(BoxLayout):
    def __init__(self, name, cur, target, on_inc, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = dp(160)
        self.padding = dp(10)
        self.spacing = dp(5)
        with self.canvas.before:
            Color(0.2,0.2,0.2,1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(pos=self._upd, size=self._upd)
        self.name = name
        self.cur = cur
        self.target = target
        self.on_inc = on_inc
        em = {"отжимания":"💪","подтягивания":"⬆️","пресс":"🔄","приседания":"🦵"}.get(name,"🏋️")
        tl = BoxLayout(orientation='horizontal', size_hint=(1,0.25))
        tl.add_widget(Label(text=f"{em} {name.capitalize()}", font_size='18sp', bold=True, halign='left', valign='middle', color=(1,1,1,1), size_hint=(0.6,1)))
        self.val_lbl = Label(text=f"[color=00FF00]{cur}[/color]", font_size='20sp', bold=True, markup=True, halign='right', valign='middle', size_hint=(0.4,1))
        tl.add_widget(self.val_lbl)
        self.add_widget(tl)
        pl = BoxLayout(orientation='horizontal', size_hint=(1,0.2), spacing=dp(5))
        self.prog = ProgressBar(max=target, value=cur, size_hint=(0.7,1))
        pl.add_widget(self.prog)
        self.pct = Label(text=f"[color=FFFF00]{min(100,int(cur/max(1,target)*100))}%[/color]", font_size='14sp', markup=True, size_hint=(0.3,1), halign='center', valign='middle')
        pl.add_widget(self.pct)
        self.add_widget(pl)
        self.add_widget(Label(text=f"[color=AAAAAA]Цель:[/color] [color=FFFFFF]{target}[/color]", markup=True, font_size='14sp', size_hint=(1,0.2), halign='center', valign='middle'))
        inc = Button(text="➕ УВЕЛИЧИТЬ ЦЕЛЬ", font_size='14sp', background_color=(0.3,0.3,0.8,1), color=(1,1,1,1), size_hint=(1,0.35))
        inc.bind(on_press=self.inc)
        self.add_widget(inc)

    def _upd(self, *a):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def inc(self, ins):
        if self.on_inc:
            self.on_inc(self.name, self.cur, self.target)


class ProgressScreen(BoxLayout):
    def __init__(self, max_vals, exercise_ai, profile, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.padding = dp(10)
        self.max_vals = max_vals.copy()
        self.cur_vals = max_vals.copy()
        self.targets = profile.current_targets.copy()
        self.ai = exercise_ai
        self.profile = profile
        top = BoxLayout(orientation='horizontal', size_hint=(1,0.1), spacing=dp(10))
        top.add_widget(Label(text="📊 ВАШ ПРОГРЕСС", font_size='20sp', bold=True, color=(0.2,0.8,1,1), size_hint=(0.4,1), halign='left', valign='middle'))
        prof_btn = Button(text="👤 ПРОФИЛЬ", font_size='14sp', background_color=(0.4,0.4,0.8,1), color=(1,1,1,1), size_hint=(0.3,1))
        prof_btn.bind(on_press=self.show_profile)
        top.add_widget(prof_btn)
        rst = Button(text="🔄 Сброс", font_size='14sp', background_color=(0.5,0.5,0.5,1), color=(1,1,1,1), size_hint=(0.3,1))
        rst.bind(on_press=self.reset_all)
        top.add_widget(rst)
        self.add_widget(top)
        sc = ScrollView(size_hint=(1,0.5))
        self.cards = BoxLayout(orientation='vertical', spacing=dp(10), size_hint_y=None)
        self.cards.bind(minimum_height=self.cards.setter('height'))
        self.cards_dict = {}
        for ex, val in max_vals.items():
            cd = ExerciseCard(ex, self.cur_vals[ex], self.targets[ex], self.on_inc)
            self.cards.add_widget(cd)
            self.cards_dict[ex] = cd
        sc.add_widget(self.cards)
        self.add_widget(sc)
        tr = BoxLayout(orientation='horizontal', size_hint=(1,0.1), spacing=dp(10))
        train = Button(text="🏋️ ТРЕНИРОВАТЬСЯ", font_size='18sp', background_color=(0.2,0.8,0.2,1), color=(1,1,1,1), size_hint=(0.7,1))
        train.bind(on_press=self.start_train)
        tr.add_widget(train)
        back = Button(text="◀️ Назад", font_size='16sp', background_color=(0.5,0.5,0.5,1), color=(1,1,1,1), size_hint=(0.3,1))
        back.bind(on_press=self.go_back)
        tr.add_widget(back)
        self.add_widget(tr)
        self.add_widget(Label(text="⚠ ДЕМО-РЕЖИМ: Счет по движению" if self.ai.demo_mode else "✅ ИИ активен", font_size='14sp', color=(0.8,0.8,0.8,1), size_hint=(1,0.05), halign='center'))
        self.add_widget(Label(text=f"📅 День {profile.total_days}  |  🔥 Серия: {profile.streak_days} дней", font_size='12sp', color=(0.8,0.8,0.8,1), size_hint=(1,0.05), halign='center'))

    def on_inc(self, ex, cur, targ):
        if self.profile.increase_target(ex):
            self.targets[ex] = self.profile.current_targets[ex]
            cd = self.cards_dict.get(ex)
            if cd:
                cd.target = self.targets[ex]
                cd.prog.max = self.targets[ex]
                pct = min(100, int(cur/max(1,self.targets[ex])*100))
                cd.pct.text = f"[color=FFFF00]{pct}%[/color]"
                cd.prog.value = cur
            App.get_running_app().show_success_popup(f"Цель для {ex} увеличена!\nНовая цель: {self.targets[ex]}")

    def reset_all(self, ins):
        for ex in self.cur_vals:
            self.cur_vals[ex] = self.max_vals[ex]
            cd = self.cards_dict.get(ex)
            if cd:
                cd.cur = self.max_vals[ex]
                cd.val_lbl.text = f"[color=00FF00]{self.max_vals[ex]}[/color]"
                pct = min(100, int(self.max_vals[ex]/max(1,self.targets[ex])*100))
                cd.pct.text = f"[color=FFFF00]{pct}%[/color]"
                cd.prog.value = self.max_vals[ex]
        print("✓ Все значения сброшены")

    def start_train(self, ins):
        App.get_running_app().show_exercise_selection(self.cur_vals)

    def show_profile(self, ins):
        App.get_running_app().show_profile_screen()

    def go_back(self, ins):
        App.get_running_app().show_max_input()


class TrainingScreen(BoxLayout):
    def __init__(self, cur_vals, exercise_ai, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.padding = dp(10)
        self.cur_vals = cur_vals
        self.ai = exercise_ai
        self.add_widget(Label(text="🏋️ ВЫБЕРИТЕ УПРАЖНЕНИЕ", font_size='24sp', bold=True, color=(0.2,0.8,1,1), size_hint=(1,0.15)))
        gr = GridLayout(cols=2, spacing=dp(15), padding=dp(15), size_hint=(1,0.7))
        for ex, em, col in [("отжимания","💪",(0.3,0.6,0.3,1)),("подтягивания","⬆️",(0.3,0.3,0.6,1)),("пресс","🔄",(0.6,0.3,0.3,1)),("приседания","🦵",(0.6,0.6,0.3,1))]:
            if ex in cur_vals:
                st = self.ai.get_model_status(ex)
                btn = Button(text=f"{em}\n{ex.capitalize()}\nТекущий: {cur_vals[ex]}\n{st}", font_size='16sp', size_hint=(1,1), background_color=col)
                btn.bind(on_press=lambda x, e=ex: self.start_ex(e))
                gr.add_widget(btn)
        self.add_widget(gr)
        back = Button(text="◀️ НАЗАД К ПРОГРЕССУ", font_size='18sp', size_hint=(1,0.1), background_color=(0.5,0.5,0.5,1))
        back.bind(on_press=self.go_back)
        self.add_widget(back)

    def start_ex(self, ex):
        App.get_running_app().start_exercise_session(ex, self.cur_vals[ex])

    def go_back(self, ins):
        App.get_running_app().show_progress_screen()


class ExerciseSessionScreen(BoxLayout):
    def __init__(self, ai, ex, target, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(5)
        self.padding = dp(5)
        self.ai = ai
        self.ex = ex
        self.target = target
        self.start_cnt = ai.get_count(ex)
        self.add_widget(Label(text=f"{ex.capitalize()} - Цель: {target}\n{ai.get_model_status(ex)}", font_size='18sp', bold=True, color=(0.2,0.6,1,1), size_hint=(1,0.1)))
        self.cam = KivyCamera(ai, ex, on_rep_callback=self.on_rep)
        self.add_widget(self.cam)
        info = BoxLayout(orientation='horizontal', size_hint=(1,0.1), spacing=dp(10))
        self.cnt_lbl = Label(text="Текущий счет: 0", font_size='18sp', color=(1,1,1,1))
        info.add_widget(self.cnt_lbl)
        self.stat_lbl = Label(text="Выполняется...", font_size='16sp', color=(1,1,1,1))
        info.add_widget(self.stat_lbl)
        self.add_widget(info)
        btns = BoxLayout(orientation='horizontal', size_hint=(1,0.1), spacing=dp(10))
        rst = Button(text="🔄 Сброс", background_color=(0.3,0.3,0.8,1), size_hint=(0.3,1))
        rst.bind(on_press=self.reset)
        btns.add_widget(rst)
        cmpl = Button(text="✅ Завершить", background_color=(0.2,0.8,0.2,1), size_hint=(0.4,1))
        cmpl.bind(on_press=self.complete)
        btns.add_widget(cmpl)
        cncl = Button(text="✖ Отмена", background_color=(0.8,0.3,0.3,1), size_hint=(0.3,1))
        cncl.bind(on_press=self.cancel)
        btns.add_widget(cncl)
        self.add_widget(btns)
        Clock.schedule_interval(self.update_info, 0.1)

    def on_rep(self):
        cur = self.ai.get_count(self.ex) - self.start_cnt
        if cur >= self.target:
            self.stat_lbl.text = "✅ Цель достигнута!"
            self.stat_lbl.color = (0,1,0,1)

    def update_info(self, dt):
        cur = self.ai.get_count(self.ex) - self.start_cnt
        self.cnt_lbl.text = f"Сделано: {cur}/{self.target}"

    def reset(self, ins):
        self.ai.reset_counts()
        self.start_cnt = 0
        self.stat_lbl.text = "Счет сброшен"
        self.stat_lbl.color = (1,1,0,1)
        Clock.schedule_once(lambda dt: setattr(self.stat_lbl, 'text', "Выполняется..."), 1)

    def complete(self, ins):
        cur = self.ai.get_count(self.ex) - self.start_cnt
        app = App.get_running_app()
        if cur > 0:
            app.update_exercise_value(self.ex, cur)
            app.add_workout_to_profile(self.ex, cur)
            app.show_success_popup(f"Вы сделали {cur} {self.ex}!")
        self.cleanup()
        app.show_training_screen()

    def cancel(self, ins):
        self.cleanup()
        App.get_running_app().show_training_screen()

    def cleanup(self):
        self.cam.stop_camera()
        Clock.unschedule(self.update_info)
        self.ai.reset_counts()


class Mobile_Trainer(App):
    def build(self):
        self.title = "Mobile Trainer PRO"
        Window.clearcolor = (0.1,0.1,0.1,1)
        self.ai = ExerciseAI()
        self.profile = UserProfile()
        self.cur_vals = {}
        self.layout = BoxLayout()
        self.show_max_input()
        return self.layout

    def show_max_input(self):
        self.layout.clear_widgets()
        self.layout.add_widget(MaxInputScreen(self.profile, self.on_max))

    def on_max(self, vals):
        self.cur_vals = vals
        self.show_progress()

    def show_progress(self):
        self.layout.clear_widgets()
        self.layout.add_widget(ProgressScreen(self.cur_vals, self.ai, self.profile))

    def show_training(self):
        self.layout.clear_widgets()
        self.layout.add_widget(TrainingScreen(self.cur_vals, self.ai))

    def show_exercise_selection(self, cur):
        self.layout.clear_widgets()
        self.layout.add_widget(TrainingScreen(cur, self.ai))

    def show_profile_screen(self):
        self.layout.clear_widgets()
        self.layout.add_widget(ProfileScreen(self.profile, self.show_progress))

    def start_exercise_session(self, ex, target):
        self.layout.clear_widgets()
        self.layout.add_widget(ExerciseSessionScreen(self.ai, ex, target))

    def update_exercise_value(self, ex, inc):
        if ex in self.cur_vals:
            self.cur_vals[ex] += inc
            print(f"✓ {ex} увеличен на {inc}, теперь {self.cur_vals[ex]}")

    def add_workout_to_profile(self, ex, reps):
        self.profile.add_workout(ex, reps)

    def show_error_popup(self, msg):
        cnt = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        cnt.add_widget(Label(text=msg, font_size='16sp'))
        btn = Button(text='OK', size_hint=(1,0.3))
        pop = Popup(title='Ошибка', content=cnt, size_hint=(0.6,0.3))
        btn.bind(on_press=pop.dismiss)
        cnt.add_widget(btn)
        pop.open()

    def show_warning_popup(self, msg):
        cnt = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        cnt.add_widget(Label(text=msg, font_size='16sp', color=(1,1,0,1)))
        btn = Button(text='Понятно', size_hint=(1,0.3))
        pop = Popup(title='Предупреждение', content=cnt, size_hint=(0.7,0.4))
        btn.bind(on_press=pop.dismiss)
        cnt.add_widget(btn)
        pop.open()

    def show_success_popup(self, msg):
        cnt = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        cnt.add_widget(Label(text=msg, font_size='16sp', color=(0,1,0,1)))
        btn = Button(text='Отлично!', size_hint=(1,0.3))
        pop = Popup(title='Успех!', content=cnt, size_hint=(0.6,0.3))
        btn.bind(on_press=pop.dismiss)
        cnt.add_widget(btn)
        pop.open()


if __name__ == "__main__":
    try:
        Mobile_Trainer().run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        input("Нажмите Enter для выхода...")
