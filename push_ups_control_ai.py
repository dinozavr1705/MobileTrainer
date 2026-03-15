import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import joblib
from datetime import datetime
import re

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


class PushupAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.labels = ['correct_pushups', 'incorrect_pushups']
        self.training_info = {}
        self.version = "7.2.0"
        self.expected_features = 80

        self.models_dir = "saved_models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.debug = True

        # Основной датасет
        self.dataset_root = r"C:\Users\Sanya\.cache\kagglehub\datasets\hasyimabdillah\workoutfitness-video\versions\5"

        # Новый датасет с правильными и неправильными отжиманиями
        self.pushup_dataset_root = r"C:\Users\Sanya\.cache\kagglehub\datasets\mohamadashrafsalama\pushup\versions\1"
        self.correct_pushup_folder = "Correct sequence"
        self.wrong_pushup_folder = "Wrong sequence"

        self.correct_folder = "push-up"
        self.incorrect_folders = [
            "pull-up",
            "squat",
            "leg-raises",
            "wrong pushups"
        ]

        # Список ключевых слов для правильных отжиманий
        self.correct_keywords = ['correct', 'good', 'proper', 'right', 'ok', 'perfect', 'excellent']
        # Список ключевых слов для неправильных отжиманий в папке push-up
        self.incorrect_keywords = ['incorrect', 'bad', 'wrong', 'fault', 'error', 'mistake', 'poor']

        self.video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")

    def debug_print(self, msg):
        if self.debug:
            print(f"🔍 [DEBUG] {msg}")

    def is_correct_pushup(self, filename):
        """Определяет по имени файла, является ли видео правильным отжиманием"""
        filename_lower = filename.lower()

        # Сначала проверяем ключевые слова для неправильных
        for keyword in self.incorrect_keywords:
            if keyword in filename_lower:
                return False

        # Затем проверяем ключевые слова для правильных
        for keyword in self.correct_keywords:
            if keyword in filename_lower:
                return True

        # Если нет ключевых слов, считаем по последней цифре (для тестирования)
        numbers = re.findall(r'\d+', filename)
        if numbers:
            last_num = int(numbers[-1])
            return last_num % 2 == 0

        return True

    def extract_pushup_features(self, frame):
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
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                features.append(area / (perimeter + 1e-6))
            else:
                features.append(0)

            while len(features) < 20:
                features.append(0)
            return np.array(features[:20])

        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return np.zeros(20)

    def process_pushup_video(self, video_path, max_frames=30, verbose=True):
        if verbose:
            print(f"\n📹 Обработка видео: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if verbose:
                print("❌ Не удалось открыть видео")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if verbose:
            print(f"  Всего кадров: {total_frames}")
        if total_frames == 0:
            if verbose:
                print("  ❌ Нет кадров в видео")
            cap.release()
            return None

        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = range(max(1, total_frames))

        if verbose:
            print(f"  Анализ {len(frame_indices)} кадров...")

        all_features = []
        iterator = tqdm(frame_indices, desc="Обработка") if verbose else frame_indices
        for idx in iterator:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                features = self.extract_pushup_features(frame)
                all_features.append(features)

        cap.release()

        if len(all_features) > 0:
            features_array = np.array(all_features)
            mean_f = np.mean(features_array, axis=0)
            std_f = np.std(features_array, axis=0)
            max_f = np.max(features_array, axis=0)
            min_f = np.min(features_array, axis=0)
            combined = np.concatenate([mean_f, std_f, max_f, min_f])
            if len(combined) != 80:
                combined = np.pad(combined, (0, max(0, 80 - len(combined))))[:80]
            return combined

        return None

    def load_dataset(self, max_frames_per_video=30, max_videos_per_class=None):
        print(f"\n📂 Загрузка датасета из: {self.dataset_root}")
        print(f"📂 И дополнительного датасета: {self.pushup_dataset_root}")

        X_list = []
        y_list = []
        paths_used = []
        stats = {
            'correct': 0,
            'incorrect': 0,
            'skipped': 0,
            'correct_in_pushup': 0,
            'incorrect_in_pushup': 0,
            'from_wrong_pushups': 0,
            'from_correct_sequence': 0,
            'from_wrong_sequence': 0
        }

        # 1. Загружаем из основного датасета (hasyimabdillah)
        if os.path.exists(self.dataset_root):
            # Загружаем правильные отжимания из папки push-up
            correct_path = os.path.join(self.dataset_root, self.correct_folder)
            if os.path.exists(correct_path):
                print(f"\n🔍 Анализ видео в {self.correct_folder}...")
                all_pushup_videos = []
                for ext in self.video_extensions:
                    all_pushup_videos.extend(glob.glob(os.path.join(correct_path, "**", "*" + ext), recursive=True))

                print(f"   Всего видео в папке: {len(all_pushup_videos)}")

                correct_in_pushup = []
                incorrect_in_pushup = []

                for video_path in all_pushup_videos:
                    filename = os.path.basename(video_path)
                    if self.is_correct_pushup(filename):
                        correct_in_pushup.append(video_path)
                    else:
                        incorrect_in_pushup.append(video_path)

                print(f"   Правильные по ключевым словам: {len(correct_in_pushup)}")
                print(f"   Неправильные по ключевым словам: {len(incorrect_in_pushup)}")

                limit = max_videos_per_class if max_videos_per_class else len(correct_in_pushup)
                for video_path in tqdm(correct_in_pushup[:limit], desc="Правильные (основной датасет)"):
                    features = self.process_pushup_video(video_path, max_frames=max_frames_per_video, verbose=False)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(0)
                        paths_used.append((video_path, 0, "correct_pushup_main"))
                        stats['correct'] += 1
                        stats['correct_in_pushup'] += 1
                    else:
                        stats['skipped'] += 1

                limit = max_videos_per_class if max_videos_per_class else len(incorrect_in_pushup)
                for video_path in tqdm(incorrect_in_pushup[:limit], desc="Неправильные (из push-up)"):
                    features = self.process_pushup_video(video_path, max_frames=max_frames_per_video, verbose=False)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(1)
                        paths_used.append((video_path, 1, "incorrect_pushup_from_pushup"))
                        stats['incorrect'] += 1
                        stats['incorrect_in_pushup'] += 1
                    else:
                        stats['skipped'] += 1

            # Загружаем из других папок основного датасета
            print(f"\n❌ Загрузка неправильных примеров из других упражнений...")
            for folder in self.incorrect_folders:
                folder_path = os.path.join(self.dataset_root, folder)
                if not os.path.exists(folder_path):
                    print(f"   ⚠️ Папка не найдена: {folder}")
                    continue

                print(f"   Поиск в {folder}...")

                incorrect_videos = []
                for ext in self.video_extensions:
                    incorrect_videos.extend(glob.glob(os.path.join(folder_path, "**", "*" + ext), recursive=True))

                print(f"      Найдено видео: {len(incorrect_videos)}")

                limit = max_videos_per_class if max_videos_per_class else len(incorrect_videos)
                source_name = "wrong_pushups" if folder == "wrong pushups" else folder

                for video_path in tqdm(incorrect_videos[:limit], desc=folder):
                    features = self.process_pushup_video(video_path, max_frames=max_frames_per_video, verbose=False)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(1)
                        paths_used.append((video_path, 1, source_name))
                        stats['incorrect'] += 1
                        if folder == "wrong pushups":
                            stats['from_wrong_pushups'] += 1
                    else:
                        stats['skipped'] += 1

        # 2. Загружаем из дополнительного датасета (mohamadashrafsalama)
        if os.path.exists(self.pushup_dataset_root):
            print(f"\n📂 Загрузка из дополнительного датасета...")

            # Правильные отжимания из Correct sequence
            correct_seq_path = os.path.join(self.pushup_dataset_root, self.correct_pushup_folder)
            if os.path.exists(correct_seq_path):
                print(f"\n✅ Загрузка правильных отжиманий из {self.correct_pushup_folder}...")
                correct_videos = []
                for ext in self.video_extensions:
                    correct_videos.extend(glob.glob(os.path.join(correct_seq_path, "**", "*" + ext), recursive=True))

                print(f"   Найдено видео: {len(correct_videos)}")

                limit = max_videos_per_class if max_videos_per_class else len(correct_videos)
                for video_path in tqdm(correct_videos[:limit], desc="Правильные (Correct sequence)"):
                    features = self.process_pushup_video(video_path, max_frames=max_frames_per_video, verbose=False)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(0)
                        paths_used.append((video_path, 0, "correct_sequence"))
                        stats['correct'] += 1
                        stats['from_correct_sequence'] += 1
                    else:
                        stats['skipped'] += 1

            # Неправильные отжимания из Wrong sequence
            wrong_seq_path = os.path.join(self.pushup_dataset_root, self.wrong_pushup_folder)
            if os.path.exists(wrong_seq_path):
                print(f"\n❌ Загрузка неправильных отжиманий из {self.wrong_pushup_folder}...")
                wrong_videos = []
                for ext in self.video_extensions:
                    wrong_videos.extend(glob.glob(os.path.join(wrong_seq_path, "**", "*" + ext), recursive=True))

                print(f"   Найдено видео: {len(wrong_videos)}")

                limit = max_videos_per_class if max_videos_per_class else len(wrong_videos)
                for video_path in tqdm(wrong_videos[:limit], desc="Неправильные (Wrong sequence)"):
                    features = self.process_pushup_video(video_path, max_frames=max_frames_per_video, verbose=False)
                    if features is not None:
                        X_list.append(features)
                        y_list.append(1)
                        paths_used.append((video_path, 1, "wrong_sequence"))
                        stats['incorrect'] += 1
                        stats['from_wrong_sequence'] += 1
                    else:
                        stats['skipped'] += 1

        print(f"\n📊 Статистика загрузки:")
        print(f"   ✅ Правильные отжимания: {stats['correct']}")
        print(f"      - из основного датасета (push-up): {stats['correct_in_pushup']}")
        print(f"      - из Correct sequence: {stats['from_correct_sequence']}")
        print(f"   ❌ Неправильные примеры: {stats['incorrect']}")
        print(f"      - из основного датасета (push-up по ключевым): {stats['incorrect_in_pushup']}")
        print(f"      - из wrong pushups: {stats['from_wrong_pushups']}")
        print(f"      - из Wrong sequence: {stats['from_wrong_sequence']}")
        print(
            f"      - из других упражнений: {stats['incorrect'] - stats['incorrect_in_pushup'] - stats['from_wrong_pushups'] - stats['from_wrong_sequence']}")
        print(f"   ⏭️  Пропущено (ошибка обработки): {stats['skipped']}")

        if len(X_list) == 0:
            print("\n❌ Не удалось загрузить ни одного видео")
            return None, None, paths_used

        return np.array(X_list), np.array(y_list), paths_used

    def build_model(self, input_dim=80):
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        return model

    def train(self, epochs=50, max_videos_per_class=100):
        print("=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ОТЖИМАНИЙ")
        print("=" * 60)

        X, y, paths = self.load_dataset(
            max_frames_per_video=30,
            max_videos_per_class=max_videos_per_class
        )

        if X is None or len(X) < 20:
            print("\n❌ Недостаточно данных для обучения")
            return False

        print(f"\n📊 Итоговый набор данных: {len(X)} примеров")
        print(f"   Размерность признаков: {X.shape}")
        print(f"   Правильных: {np.sum(y == 0)}")
        print(f"   Неправильных: {np.sum(y == 1)}")

        if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
            print("\n❌ Один из классов отсутствует. Обучение невозможно.")
            return False

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
        ]

        print("\n📋 Архитектура модели:")
        self.model.summary()

        print("\n🚀 Начало обучения...")
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=16,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        print("\n📊 Оценка модели...")
        test_results = self.model.evaluate(X_test_scaled, y_test, verbose=0)

        print(f"\n📈 РЕЗУЛЬТАТЫ:")
        print(f"   Точность (accuracy): {test_results[1]:.3f}")
        if len(test_results) > 2:
            print(f"   Precision: {test_results[2]:.3f}")
            print(f"   Recall: {test_results[3]:.3f}")
        print(f"   Потеря (loss): {test_results[0]:.3f}")

        y_pred_prob = self.model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        unique_pred = np.unique(y_pred)
        unique_true = np.unique(y_test)

        if len(unique_true) == 2 and len(unique_pred) == 2:
            print("\n📋 Отчет классификации:")
            print(classification_report(y_test, y_pred, target_names=self.labels))
            print("\n📊 Матрица ошибок:")
            print(confusion_matrix(y_test, y_pred))

            errors = np.where(y_pred != y_test)[0]
            if len(errors) > 0:
                print(f"\n⚠️ Ошибок на тесте: {len(errors)}")
                print("   Примеры видео, которые модель путает:")
                for i, idx in enumerate(errors[:3]):
                    true_label = "правильные" if y_test[idx] == 0 else "неправильные"
                    pred_label = "правильные" if y_pred[idx] == 0 else "неправильные"
                    print(
                        f"   {i + 1}. {os.path.basename(paths[idx][0])} - должно быть {true_label}, модель думает {pred_label}")
        else:
            print(f"\n📊 Точность на тесте: {np.mean(y_pred == y_test):.3f}")

        self.plot_training_history(history)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.models_dir, f"pushup_model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'model.keras')
        self.model.save(model_path)

        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        keywords_info = {
            'correct_keywords': self.correct_keywords,
            'incorrect_keywords': self.incorrect_keywords
        }
        keywords_path = os.path.join(model_dir, 'keywords.json')
        with open(keywords_path, 'w') as f:
            json.dump(keywords_info, f, indent=2)

        self.training_info = {
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_samples': len(X),
            'num_correct': int(np.sum(y == 0)),
            'num_incorrect': int(np.sum(y == 1)),
            'test_accuracy': float(test_results[1]),
            'exercise_type': 'pushups',
            'model_dir': model_dir,
            'dataset_root': self.dataset_root,
            'pushup_dataset_root': self.pushup_dataset_root,
            'correct_keywords': self.correct_keywords,
            'incorrect_keywords': self.incorrect_keywords
        }

        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.training_info, f, indent=2)

        self.is_trained = True

        print(f"\n✅ Модель сохранена в: {model_dir}")
        print(f"\n📝 Ключевые слова для правильных отжиманий: {', '.join(self.correct_keywords)}")
        print(f"📝 Ключевые слова для неправильных отжиманий: {', '.join(self.incorrect_keywords)}")

        return True

    def plot_training_history(self, history):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Train', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        plt.title('Точность модели', fontsize=14)
        plt.xlabel('Эпоха')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Train', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
        plt.title('Потери модели', fontsize=14)
        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        if 'precision' in history.history:
            plt.plot(history.history['precision'], label='Train Precision', linewidth=2)
            plt.plot(history.history['val_precision'], label='Val Precision', linewidth=2)
        if 'recall' in history.history:
            plt.plot(history.history['recall'], label='Train Recall', linewidth=2, linestyle='--')
            plt.plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
        plt.title('Precision/Recall', fontsize=14)
        plt.xlabel('Эпоха')
        plt.ylabel('Значение')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pushup_training_history.png', dpi=150)
        plt.show()

    def analyze_video(self, video_path):
        if not self.is_trained:
            print("❌ Модель не обучена!")
            return None

        print("\n" + "=" * 60)
        print(f"🔍 АНАЛИЗ ВИДЕО: {os.path.basename(video_path)}")
        print("=" * 60)

        features = self.process_pushup_video(video_path, verbose=True)

        if features is None:
            print("\n❌ Не удалось извлечь признаки")
            return None

        if self.scaler is not None:
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = [features]

        prediction_prob = self.model.predict(features_scaled, verbose=0)[0][0]

        print(f"\n📊 РЕЗУЛЬТАТ:")
        print(f"  Вероятность правильности: {prediction_prob:.3f}")

        if prediction_prob > 0.5:
            print(f"  ✅ ПРАВИЛЬНЫЕ отжимания")
            print(f"  Уверенность: {prediction_prob:.1%}")
        else:
            print(f"  ❌ НЕПРАВИЛЬНЫЕ отжимания")
            print(f"  Уверенность: {1 - prediction_prob:.1%}")

        return {
            'prediction': 'correct' if prediction_prob > 0.5 else 'incorrect',
            'confidence': float(prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob),
            'probability': float(prediction_prob)
        }

    def save_model(self, filename="pushup_model.keras"):
        if not self.is_trained:
            print("Модель не обучена!")
            return False

        try:
            self.model.save(filename)
            print(f"✅ Модель сохранена: {filename}")

            if self.scaler is not None:
                scaler_filename = filename.replace('.keras', '_scaler.pkl')
                joblib.dump(self.scaler, scaler_filename)
                print(f"✅ Scaler сохранен: {scaler_filename}")

            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False

    def load_model(self, filename="pushup_model.keras"):
        try:
            if os.path.isdir(filename):
                filename = os.path.join(filename, "model.keras")
            if not os.path.exists(filename):
                print(f"❌ Файл не найден: {filename}")
                return False
            print(f"\n📂 Загрузка модели из: {filename}")
            self.model = keras.models.load_model(filename)
            print("✅ Модель загружена")

            model_dir = os.path.dirname(os.path.abspath(filename))
            scaler_in_dir = os.path.join(model_dir, "scaler.pkl")
            scaler_named = filename.replace(".keras", "_scaler.pkl")
            if os.path.exists(scaler_in_dir):
                self.scaler = joblib.load(scaler_in_dir)
                print("✅ Scaler загружен (scaler.pkl)")
            elif os.path.exists(scaler_named):
                self.scaler = joblib.load(scaler_named)
                print("✅ Scaler загружен (_scaler.pkl)")
            else:
                print("⚠ Scaler не найден — предсказания без нормализации")

            keywords_path = os.path.join(model_dir, 'keywords.json')
            if os.path.exists(keywords_path):
                with open(keywords_path, 'r') as f:
                    keywords = json.load(f)
                self.correct_keywords = keywords.get('correct_keywords', self.correct_keywords)
                self.incorrect_keywords = keywords.get('incorrect_keywords', self.incorrect_keywords)
                print("✅ Ключевые слова загружены")

            self.is_trained = True
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False

    def test_specific_video(self, video_path, expected_label=None):
        print("\n" + "=" * 60)
        print(f"🔍 ТЕСТИРОВАНИЕ ВИДЕО: {os.path.basename(video_path)}")
        print("=" * 60)

        if expected_label is not None:
            print(f"📌 Ожидается: {'правильные' if expected_label == 0 else 'неправильные'} отжимания")

        result = self.analyze_video(video_path)

        if result and expected_label is not None:
            is_correct = (result['prediction'] == 'correct' and expected_label == 0) or \
                         (result['prediction'] == 'incorrect' and expected_label == 1)

            if is_correct:
                print(f"\n✅ Модель угадала правильно!")
            else:
                print(f"\n❌ Модель ошиблась!")

        return result


def main():
    print("=" * 60)
    print(" АНАЛИЗАТОР ОТЖИМАНИЙ v7.2 ".center(60))
    print("=" * 60)

    analyzer = PushupAnalyzer()

    while True:
        print("\n" + "=" * 60)
        print(" МЕНЮ ".center(60))
        print("=" * 60)
        print("1. 🚀 Обучить модель")
        print("2. 📂 Загрузить модель")
        print("3. 🎥 Анализировать видео")
        print("4. 💾 Сохранить модель")
        print("5. 🔍 Проверить наличие датасета")
        print("6. 🧪 Тестировать конкретное видео (с ожиданием)")
        print("7. 🚪 Выход")
        print("=" * 60)

        choice = input("\nВыберите действие (1-7): ").strip()

        if choice == "1":
            epochs = input("Введите количество эпох (по умолчанию 50): ").strip()
            epochs = int(epochs) if epochs else 50

            max_videos = input("Максимум видео на класс (по умолчанию 100): ").strip()
            max_videos = int(max_videos) if max_videos else 100

            print("\n📝 Текущие ключевые слова для правильных отжиманий:")
            print(f"   {', '.join(analyzer.correct_keywords)}")
            change = input("Изменить? (д/н): ").strip().lower()
            if change in ['д', 'да', 'y', 'yes']:
                new_keywords = input("Введите ключевые слова через запятую: ").strip()
                if new_keywords:
                    analyzer.correct_keywords = [k.strip() for k in new_keywords.split(',')]

            print("\n📝 Текущие ключевые слова для неправильных отжиманий:")
            print(f"   {', '.join(analyzer.incorrect_keywords)}")
            change = input("Изменить? (д/н): ").strip().lower()
            if change in ['д', 'да', 'y', 'yes']:
                new_keywords = input("Введите ключевые слова через запятую: ").strip()
                if new_keywords:
                    analyzer.incorrect_keywords = [k.strip() for k in new_keywords.split(',')]

            analyzer.train(epochs=epochs, max_videos_per_class=max_videos)

        elif choice == "2":
            filename = input("Введите имя файла модели или папки: ").strip()
            if filename:
                analyzer.load_model(filename)

        elif choice == "3":
            if not analyzer.is_trained:
                print("❌ Сначала обучите или загрузите модель!")
                continue

            video_path = input("Введите путь к видео: ").strip()
            if os.path.exists(video_path):
                analyzer.analyze_video(video_path)
            else:
                print("❌ Файл не найден")

        elif choice == "4":
            if analyzer.is_trained:
                filename = input("Введите имя файла: ").strip()
                if not filename:
                    filename = "pushup_model.keras"
                analyzer.save_model(filename)
            else:
                print("❌ Сначала обучите модель!")

        elif choice == "5":
            print(f"\n🔍 Основной датасет: {analyzer.dataset_root}")
            print(f"🔍 Дополнительный датасет: {analyzer.pushup_dataset_root}")

            if os.path.exists(analyzer.dataset_root):
                print("✅ Основной датасет найден")

                correct_path = os.path.join(analyzer.dataset_root, analyzer.correct_folder)
                if os.path.exists(correct_path):
                    videos = []
                    for ext in analyzer.video_extensions:
                        videos.extend(glob.glob(os.path.join(correct_path, "**", "*" + ext), recursive=True))
                    print(f"   📁 {analyzer.correct_folder}: {len(videos)} видео")

                for folder in analyzer.incorrect_folders:
                    folder_path = os.path.join(analyzer.dataset_root, folder)
                    if os.path.exists(folder_path):
                        videos = []
                        for ext in analyzer.video_extensions:
                            videos.extend(glob.glob(os.path.join(folder_path, "**", "*" + ext), recursive=True))
                        print(f"   📁 {folder}: {len(videos)} видео")
            else:
                print("❌ Основной датасет не найден")

            if os.path.exists(analyzer.pushup_dataset_root):
                print("✅ Дополнительный датасет найден")

                correct_seq = os.path.join(analyzer.pushup_dataset_root, analyzer.correct_pushup_folder)
                if os.path.exists(correct_seq):
                    videos = []
                    for ext in analyzer.video_extensions:
                        videos.extend(glob.glob(os.path.join(correct_seq, "**", "*" + ext), recursive=True))
                    print(f"   📁 {analyzer.correct_pushup_folder}: {len(videos)} видео")

                wrong_seq = os.path.join(analyzer.pushup_dataset_root, analyzer.wrong_pushup_folder)
                if os.path.exists(wrong_seq):
                    videos = []
                    for ext in analyzer.video_extensions:
                        videos.extend(glob.glob(os.path.join(wrong_seq, "**", "*" + ext), recursive=True))
                    print(f"   📁 {analyzer.wrong_pushup_folder}: {len(videos)} видео")
            else:
                print("❌ Дополнительный датасет не найден")

        elif choice == "6":
            if not analyzer.is_trained:
                print("❌ Сначала обучите или загрузите модель!")
                continue

            video_path = input("Введите путь к видео для тестирования: ").strip()
            if not os.path.exists(video_path):
                print("❌ Файл не найден")
                continue

            expected = input("Ожидается правильное отжимание? (д/н): ").strip().lower()
            expected_label = 0 if expected in ['д', 'да', 'y', 'yes'] else 1

            analyzer.test_specific_video(video_path, expected_label)

        elif choice == "7":
            print("\n👋 Выход...")
            break

        else:
            print("❌ Неверный выбор!")


if __name__ == "__main__":
    main()