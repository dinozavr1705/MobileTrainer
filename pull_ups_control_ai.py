import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import kagglehub
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

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


class PullupAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.labels = ['correct', 'incorrect']
        self.training_info = {}
        self.version = "1.1.0"
        self.expected_features = 72

    def is_likely_pullup(self, frame):
        """Проверяет, может ли кадр быть подтягиванием"""
        try:
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            height, width = gray.shape

            # Для подтягиваний характерны:
            # 1. Высокое соотношение сторон (вертикальная ориентация)
            aspect_ratio = height / width

            # 2. Много вертикальных линий (перекладина, тело)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Вертикальные градиенты сильнее горизонтальных для подтягиваний
            vertical_strength = np.mean(np.abs(sobely))
            horizontal_strength = np.mean(np.abs(sobelx))
            vertical_ratio = vertical_strength / (horizontal_strength + 1e-6)

            # 3. Темная область вверху (перекладина)
            top_region = gray[:height // 4, :]
            bottom_region = gray[3 * height // 4:, :]
            top_darkness = np.mean(top_region)
            bottom_darkness = np.mean(bottom_region)
            darkness_ratio = top_darkness / (bottom_darkness + 1e-6)

            # Критерии для подтягиваний:
            is_pullup_likely = (
                    aspect_ratio > 1.2 and  # Более вертикальное
                    vertical_ratio > 1.5 and  # Вертикальные линии преобладают
                    darkness_ratio < 0.8  # Верх темнее низа
            )

            confidence = 0
            if aspect_ratio > 1.3:
                confidence += 0.3
            if vertical_ratio > 1.5:
                confidence += 0.3
            if darkness_ratio < 0.7:
                confidence += 0.4

            return is_pullup_likely, confidence

        except Exception as e:
            print(f"Ошибка проверки типа упражнения: {e}")
            return False, 0

    def extract_pullup_features(self, frame):
        """Специфичные признаки для подтягиваний"""
        try:
            # Для подтягиваний важно вертикальное положение
            frame = cv2.resize(frame, (320, 480))  # Вертикальный формат
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            features = []

            # 1. Вертикальные характеристики (3 признака)
            height, width = gray.shape
            features.append(height / width)  # Соотношение сторон

            # 2. Зональный анализ (3 признака)
            top_zone = gray[:height // 3, :]
            middle_zone = gray[height // 3:2 * height // 3, :]
            bottom_zone = gray[2 * height // 3:, :]

            features.append(np.mean(top_zone))
            features.append(np.mean(middle_zone))
            features.append(np.mean(bottom_zone))

            # 3. Контраст верха/низа (1 признак)
            features.append(np.mean(top_zone) - np.mean(bottom_zone))

            # 4. Признаки для перекладины (2 признака)
            edges = cv2.Canny(gray, 50, 150)
            top_edges = edges[:height // 4, :]
            features.append(np.mean(top_edges) / 255)

            # Горизонтальные линии в верхней части
            horizontal_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            top_horizontal = cv2.filter2D(top_zone, -1, horizontal_kernel)
            features.append(np.mean(top_horizontal))

            # 5. Вертикальные градиенты (2 признака)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            features.append(np.mean(sobely))
            features.append(np.std(sobely))

            # 6. Кожные тона (руки) - меньше для подтягиваний (1 признак)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            features.append(np.mean(skin_mask) / 255)

            # 7. Вертикальная симметрия (1 признак)
            left_half = gray[:, :width // 2]
            right_half = gray[:, width // 2:]
            if left_half.shape == right_half.shape:
                vertical_symmetry = np.mean(np.abs(left_half - right_half))
            else:
                vertical_symmetry = 0
            features.append(vertical_symmetry)

            # 8. Плотность объектов (3 признака - по зонам)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Верхняя зона
            top_binary = thresh[:height // 3, :]
            features.append(np.mean(top_binary) / 255)

            # Средняя зона
            middle_start = height // 3
            middle_end = 2 * height // 3
            if middle_end <= thresh.shape[0]:
                middle_binary = thresh[middle_start:middle_end, :]
                features.append(np.mean(middle_binary) / 255)
            else:
                features.append(0)

            # Нижняя зона
            bottom_start = 2 * height // 3
            if bottom_start < thresh.shape[0]:
                bottom_binary = thresh[bottom_start:, :]
                features.append(np.mean(bottom_binary) / 255)
            else:
                features.append(0)

            # 9. Контраст и энтропия (2 признака)
            features.append(np.max(gray) - np.min(gray))

            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            features.append(float(entropy))

            # 10. Отношение вертикальных/горизонтальных градиентов (1 признак)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            vertical_ratio = np.mean(np.abs(sobely)) / (np.mean(np.abs(sobelx)) + 1e-6)
            features.append(vertical_ratio)

            # Проверяем что у нас ровно 18 признаков
            if len(features) != 18:
                # Заполняем нулями до 18 или обрезаем
                features = features[:18] if len(features) > 18 else features + [0] * (18 - len(features))

            return np.array(features)

        except Exception as e:
            print(f"Ошибка извлечения признаков: {e}")
            return np.zeros(18)

    def process_pullup_video(self, video_path, max_frames=30):
        """Обработка видео с подтягиваниями"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return None

        features_per_frame = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Проверяем первые несколько кадров на подтягивания
        pullup_frames = 0
        total_checked = 0

        for i in range(min(10, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                is_pullup, confidence = self.is_likely_pullup(frame)
                if is_pullup:
                    pullup_frames += 1
                total_checked += 1

        # Если менее 30% кадров похожи на подтягивания, вероятно это не они
        if total_checked > 0 and (pullup_frames / total_checked) < 0.3:
            print(f"⚠️  Только {pullup_frames}/{total_checked} кадров похожи на подтягивания")
            print("Возможно, это не подтягивания!")
            cap.release()
            return None

        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = range(total_frames)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                features = self.extract_pullup_features(frame)
                if features is not None and len(features) == 18:
                    features_per_frame.append(features)

        cap.release()

        if len(features_per_frame) > 0:
            features_array = np.array(features_per_frame)

            # 4 статистики по 18 признакам = 72 признака
            mean_features = np.mean(features_array, axis=0)
            std_features = np.std(features_array, axis=0)
            max_features = np.max(features_array, axis=0)
            min_features = np.min(features_array, axis=0)

            combined_features = np.concatenate([
                mean_features,
                std_features,
                max_features,
                min_features
            ])

            # Корректируем до 72 признаков если нужно
            if len(combined_features) != 72:
                if len(combined_features) < 72:
                    combined_features = np.pad(combined_features, (0, 72 - len(combined_features)))
                else:
                    combined_features = combined_features[:72]

            return combined_features

        return None

    def load_pullup_dataset(self):
        """Загрузка данных для подтягиваний - ищем реальные видео"""
        print("Поиск реальных видео с подтягиваниями...")

        # Ищем видео с подтягиваниями в текущей директории
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        all_videos = []

        for ext in video_extensions:
            all_videos.extend(glob.glob(ext))

        print(f"Найдено {len(all_videos)} видеофайлов в текущей директории")

        video_data = []
        pullup_videos_found = 0

        for video_path in all_videos[:20]:  # Проверяем первые 20 видео
            path_lower = video_path.lower()

            # Проверяем по названию
            is_named_pullup = any(word in path_lower for word in [
                'pullup', 'pull-up', 'chinup', 'chinning', 'подтяг'
            ])

            # Также проверяем первые кадры
            cap = cv2.VideoCapture(video_path)
            is_visual_pullup = False
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    is_visual_pullup, _ = self.is_likely_pullup(frame)
                cap.release()

            if is_named_pullup or is_visual_pullup:
                print(f"Найдено видео с подтягиваниями: {os.path.basename(video_path)}")
                pullup_videos_found += 1

                features = self.process_pullup_video(video_path)
                if features is not None and len(features) == 72:
                    # Определяем правильность по названию
                    label = 0  # По умолчанию правильные
                    if any(word in path_lower for word in ['incorrect', 'bad', 'wrong', 'faulty', 'неправ']):
                        label = 1
                    elif any(word in path_lower for word in ['correct', 'good', 'proper', 'прав']):
                        label = 0
                    else:
                        # Случайно для разнообразия
                        label = np.random.randint(0, 2)

                    video_data.append({
                        'path': video_path,
                        'features': features,
                        'label': label
                    })

        print(f"\nНайдено {pullup_videos_found} видео с подтягиваниями")

        # Если реальных данных мало, создаем синтетические
        if len(video_data) < 30:
            print(f"Мало реальных данных ({len(video_data)}). Создаю синтетические...")
            synthetic_data = self.create_synthetic_data(60 - len(video_data))
            video_data.extend(synthetic_data)

        print(f"\nИтого: {len(video_data)} примеров подтягиваний")
        print(f"  Правильные: {len([v for v in video_data if v['label'] == 0])}")
        print(f"  Неправильные: {len([v for v in video_data if v['label'] == 1])}")

        return video_data

    def create_synthetic_data(self, num_samples):
        """Создание синтетических данных, похожих на подтягивания"""
        synthetic_data = []

        for i in range(num_samples):
            # Характеристики подтягиваний
            if i % 2 == 0:  # Правильные подтягивания
                base_features = [
                    1.6,  # Высокое соотношение сторон (вертикальное)
                    60,  # Верх темный (перекладина)
                    120,  # Середина
                    180,  # Низ светлый
                    -120,  # Сильный контраст верха/низа
                    0.4,  # Много краев вверху
                    50,  # Горизонтальные линии
                    80,  # Сильные вертикальные градиенты
                    25,  # Разброс градиентов
                    0.2,  # Мало кожи (руки далеко)
                    15,  # Хорошая симметрия
                    0.3,  # Плотность верха
                    0.4,  # Плотность середины
                    0.2,  # Плотность низа
                    200,  # Высокий контраст
                    4.0,  # Высокая энтропия
                    2.5  # Вертикальные линии преобладают
                ]
                label = 0
            else:  # Неправильные подтягивания
                base_features = [
                    1.2,  # Более квадратное
                    100,  # Верх
                    100,  # Середина
                    100,  # Низ
                    0,  # Маленький контраст
                    0.1,  # Мало краев
                    10,  # Мало горизонтальных линий
                    30,  # Слабые градиенты
                    40,  # Большой разброс
                    0.4,  # Больше кожи (ближе к камере)
                    30,  # Плохая симметрия
                    0.2,  # Плотность
                    0.2,  # Плотность
                    0.2,  # Плотность
                    100,  # Низкий контраст
                    2.5,  # Низкая энтропия
                    1.0  # Равномерные градиенты
                ]
                label = 1

            # Добавляем шум
            base_features = np.array(base_features) + np.random.normal(0, 10, 17)

            # Добавляем 18-й признак (равномерность)
            base_features = np.append(base_features, np.random.uniform(0, 1))

            # Создаем 4 статистики (72 признака)
            mean_features = base_features
            std_features = np.abs(base_features) * 0.15 + 1.0
            max_features = base_features + np.abs(base_features) * 0.25
            min_features = base_features - np.abs(base_features) * 0.25

            combined_features = np.concatenate([
                mean_features,
                std_features,
                max_features,
                min_features
            ])

            synthetic_data.append({
                'path': f'synthetic_pullup_{i}',
                'features': combined_features,
                'label': label
            })

        return synthetic_data

    def build_pullup_model(self, input_dim):
        """Специальная модель для подтягиваний"""
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, epochs=25):
        """Обучение модели для подтягиваний"""
        print("=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ПОДТЯГИВАНИЙ")
        print("=" * 60)
        print("ВАЖНО: Модель обучается отличать ПРАВИЛЬНЫЕ от НЕПРАВИЛЬНЫХ подтягиваний.")
        print("Для анализа нужно использовать видео именно с ПОДТЯГИВАНИЯМИ.")
        print("=" * 60)

        # Загрузка данных
        video_data = self.load_pullup_dataset()

        if len(video_data) < 20:
            print("Недостаточно данных для обучения!")
            return False

        X = np.array([v['features'] for v in video_data])
        y = np.array([v['label'] for v in video_data])

        print(f"\nДанные: {X.shape[0]} примеров, {X.shape[1]} признаков")

        # Проверяем размерность
        if X.shape[1] != 72:
            print(f"Корректирую размерность до 72...")
            if X.shape[1] < 72:
                X = np.pad(X, ((0, 0), (0, 72 - X.shape[1])))
            else:
                X = X[:, :72]

        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Создание модели
        self.model = self.build_pullup_model(X_train.shape[1])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        # Обучение
        print("\nНачало обучения...")
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=16,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Оценка
        print("\nОценка модели...")
        test_loss, test_acc = self.model.evaluate(X_test_scaled, y_test, verbose=0)

        print(f"\n📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        print(f"  Точность на тестовых данных: {test_acc:.3f}")
        print(f"  Потеря: {test_loss:.3f}")

        # Детальная оценка
        y_pred_prob = self.model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        print("\nОтчет классификации:")
        print(classification_report(y_test, y_pred, target_names=self.labels))

        print("\nМатрица ошибок:")
        print(confusion_matrix(y_test, y_pred))

        # Визуализация
        self.plot_training_history(history)

        # Сохранение информации
        self.training_info = {
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_samples': len(video_data),
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'exercise_type': 'pullup'
        }

        self.is_trained = True

        print("\n✅ Модель для подтягиваний обучена!")
        print("⚠️  ВАЖНО: Эта модель анализирует только ПОДТЯГИВАНИЯ.")
        print("   Для отжиманий используйте другую модель.")
        return True

    def plot_training_history(self, history):
        """Визуализация обучения"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        plt.title('Model Accuracy - Pullups', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
        plt.title('Model Loss - Pullups', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pullup_training_history.png', dpi=150)
        plt.show()

    def analyze_video(self, video_path):
        """Анализ видео - проверяет, что это подтягивания перед анализом"""
        if not self.is_trained:
            print("Модель не обучена! Сначала обучите модель.")
            return None

        print(f"\n🔍 Анализ видео: {os.path.basename(video_path)}")

        # Сначала проверяем, похоже ли видео на подтягивания
        print("Проверка типа упражнения...")
        cap = cv2.VideoCapture(video_path)
        pullup_frames = 0
        total_checked = 0

        for i in range(0, 100, 10):  # Проверяем 10 кадров
            if i >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                is_pullup, confidence = self.is_likely_pullup(frame)
                if is_pullup:
                    pullup_frames += 1
                total_checked += 1

        cap.release()

        if total_checked > 0:
            pullup_ratio = pullup_frames / total_checked
            print(f"  Кадров похожих на подтягивания: {pullup_frames}/{total_checked} ({pullup_ratio:.1%})")

            if pullup_ratio < 0.5:
                print(f"\n❌ ВНИМАНИЕ: Это, вероятно, НЕ подтягивания!")
                print("   Модель анализирует только подтягивания.")
                print("   Для отжиманий используйте другую модель.")
                return None

        # Если похоже на подтягивания, анализируем
        features = self.process_pullup_video(video_path)
        if features is None:
            print("Не удалось обработать видео")
            return None

        # Корректируем размерность
        if len(features) != 72:
            print(f"Корректирую размерность признаков...")
            features = features[:72] if len(features) > 72 else np.pad(features, (0, 72 - len(features)))

        # Нормализация
        if self.scaler is not None:
            features = self.scaler.transform([features])

        # Предсказание
        prediction_prob = self.model.predict(features, verbose=0)[0][0]

        # Результат
        if prediction_prob > 0.5:
            result = "✅ ПРАВИЛЬНЫЕ подтягивания"
            confidence = prediction_prob
            emoji = "✅"
        else:
            result = "❌ НЕПРАВИЛЬНЫЕ подтягивания"
            confidence = 1 - prediction_prob
            emoji = "❌"

        print(f"\n📊 РЕЗУЛЬТАТ АНАЛИЗА ПОДТЯГИВАНИЙ:")
        print(f"  {emoji} {result}")
        print(f"  Уверенность: {confidence:.1%}")

        # Рекомендации
        if prediction_prob > 0.5:
            print(f"\n💪 ОТЛИЧНО! Подтягивания выполняются правильно.")
            print("   Рекомендации:")
            print("   - Сохраняйте прямое положение тела")
            print("   - Подтягивайтесь до уровня подбородка")
            print("   - Контролируйте опускание")
        else:
            print(f"\n⚠️  ЕСТЬ ПРОБЛЕМЫ! Подтягивания выполняются неправильно.")
            print("   Возможные проблемы:")
            print("   - Неполная амплитуда движения")
            print("   - Раскачивание тела")
            print("   - Сгибание ног")
            print("   - Быстрое опускание")

        return {
            'prediction': 'correct' if prediction_prob > 0.5 else 'incorrect',
            'confidence': float(confidence),
            'probability_correct': float(prediction_prob),
            'probability_incorrect': float(1 - prediction_prob),
            'exercise_type': 'pullup',
            'is_likely_pullup': pullup_ratio > 0.5 if total_checked > 0 else None
        }

    def save_model(self, filename="pullup_model.keras"):
        """Сохранение модели"""
        if not self.is_trained:
            print("Модель не обучена!")
            return False

        try:
            self.model.save(filename)

            if self.scaler is not None:
                scaler_filename = filename.replace('.keras', '_scaler.pkl')
                joblib.dump(self.scaler, scaler_filename)

            metadata = {
                'version': self.version,
                'training_info': self.training_info,
                'labels': self.labels
            }

            metadata_filename = filename.replace('.keras', '_metadata.json')
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✅ Модель сохранена: {filename}")
            print(f"   Для анализа используйте только видео с ПОДТЯГИВАНИЯМИ")

            return True

        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False

    def load_model(self, filename="pullup_model.keras"):
        """Загрузка модели"""
        try:
            self.model = keras.models.load_model(filename)

            scaler_filename = filename.replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_filename):
                self.scaler = joblib.load(scaler_filename)

            metadata_filename = filename.replace('.keras', '_metadata.json')
            if os.path.exists(metadata_filename):
                with open(metadata_filename, 'r') as f:
                    metadata = json.load(f)
                self.training_info = metadata.get('training_info', {})
                self.labels = metadata.get('labels', ['correct', 'incorrect'])

            self.is_trained = True
            print(f"✅ Модель загружена: {filename}")
            print(f"   Модель анализирует только ПОДТЯГИВАНИЯ")

            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False


def main():
    """Главная функция"""
    print("=" * 60)
    print(" АНАЛИЗАТОР ПОДТЯГИВАНИЙ ".center(60))
    print("=" * 60)
    print("⚠️  ВНИМАНИЕ: Эта программа анализирует только ПОДТЯГИВАНИЯ")
    print("   Для отжиманий используйте другую программу")
    print("=" * 60)

    analyzer = PullupAnalyzer()

    while True:
        print("\n" + "=" * 60)
        print(" МЕНЮ ".center(60))
        print("=" * 60)
        print("1. Обучить модель для подтягиваний")
        print("2. Загрузить сохраненную модель")
        print("3. Проанализировать видео (ТОЛЬКО подтягивания)")
        print("4. Проверить, похоже ли видео на подтягивания")
        print("5. Сохранить модель")
        print("6. Выход")
        print("=" * 60)

        choice = input("\nВыберите действие (1-6): ").strip()

        if choice == "1":
            analyzer.train(epochs=25)

        elif choice == "2":
            filename = input("Введите имя файла модели: ").strip()
            if not filename:
                filename = "pullup_model.keras"
            analyzer.load_model(filename)

        elif choice == "3":
            if not analyzer.is_trained:
                print("Сначала обучите или загрузите модель!")
                continue

            # Поиск видеофайлов
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(glob.glob(ext))

            if video_files:
                print("\nНайдены видеофайлы:")
                for i, video in enumerate(video_files[:5]):
                    print(f"  {i + 1}. {os.path.basename(video)}")
                print("  0. Ввести путь вручную")

                selection = input("\nВыберите номер: ").strip()

                if selection == "0":
                    video_path = input("Введите путь к видео: ").strip()
                else:
                    try:
                        idx = int(selection) - 1
                        if 0 <= idx < len(video_files):
                            video_path = video_files[idx]
                        else:
                            print("Неверный номер")
                            continue
                    except:
                        video_path = selection
            else:
                video_path = input("Введите путь к видеофайлу: ").strip()

            if os.path.exists(video_path):
                analyzer.analyze_video(video_path)
            else:
                print(f"Файл не найден: {video_path}")

        elif choice == "4":
            # Проверка типа упражнения
            video_path = input("Введите путь к видеофайлу: ").strip()
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                pullup_frames = 0
                total_checked = 0

                print(f"\nПроверка видео: {os.path.basename(video_path)}")

                for i in range(0, 100, 10):
                    if i >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        is_pullup, confidence = analyzer.is_likely_pullup(frame)
                        total_checked += 1
                        if is_pullup:
                            pullup_frames += 1
                        print(
                            f"  Кадр {i}: {'✅ Подтягивание' if is_pullup else '❌ Не похоже'} (уверенность: {confidence:.1%})")

                cap.release()

                if total_checked > 0:
                    ratio = pullup_frames / total_checked
                    print(f"\n📊 ИТОГ: {pullup_frames}/{total_checked} кадров похожи на подтягивания ({ratio:.1%})")

                    if ratio > 0.7:
                        print("✅ Скорее всего, это ПОДТЯГИВАНИЯ")
                    elif ratio > 0.4:
                        print("⚠️  Возможно, это подтягивания, но нужна проверка")
                    else:
                        print("❌ Скорее всего, это НЕ подтягивания")
            else:
                print(f"Файл не найден: {video_path}")

        elif choice == "5":
            if analyzer.is_trained:
                filename = input("Введите имя файла: ").strip()
                if not filename:
                    filename = "pullup_model.keras"
                analyzer.save_model(filename)
            else:
                print("Сначала обучите модель!")

        elif choice == "6":
            print("\nВыход...")
            break

        else:
            print("Неверный выбор!") 


if __name__ == "__main__":
    main()