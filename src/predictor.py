import os
import json
import logging
import tempfile
import shutil
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DVMHPerformancePredictor:
    """Класс для предсказания эффективности DVMH программ"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Инициализация предиктора
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")
        self.config = self._load_config(config_path)
        
        # Настройки из конфига
        model_config = self.config.get('model_training', {})
        self.models_dir = model_config.get('models_directory', './models')
        
        # Настройка устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация компонентов модели
        self.model = None
        self.model_metadata = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.model_type = None
        
        # Импортируем и инициализируем существующие классы
        from .data_collector import DVMHDataCollector
        from .feature_extractor import DVMHFeatureSpaceCreator
        from .model_trainer import DVMHEfficiencyMLP, DVMHAttentionModel
        
        self.data_collector = DVMHDataCollector(config_path)
        self.feature_extractor = DVMHFeatureSpaceCreator(config_path=config_path)
        self.DVMHEfficiencyMLP = DVMHEfficiencyMLP
        self.DVMHAttentionModel = DVMHAttentionModel
        
        self.logger.info(f"DVMHPerformancePredictor инициализирован, устройство: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Конфигурационный файл не найден: {config_path}")
            return {}
    
    def load_model(self, model_name: str, model_type: str = 'mlp') -> bool:
        """
        Загрузка обученной модели с препроцессорами
        
        Args:
            model_name: Имя модели (без расширения .pt)
            model_type: Тип модели ('mlp' или 'attention')
            
        Returns:
            bool: True если модель загружена успешно
        """
        self.logger.info(f"Загрузка модели: {model_name} (тип: {model_type})")
        
        # Пути к файлам модели
        model_path = os.path.join(self.models_dir, f"{model_name}.pt")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        # Проверяем существование файлов
        if not os.path.exists(model_path):
            self.logger.error(f"Файл модели не найден: {model_path}")
            return False
        
        if not os.path.exists(metadata_path):
            self.logger.error(f"Файл метаданных не найден: {metadata_path}")
            return False
        
        try:
            # Загружаем метаданные
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.model_metadata = json.load(f)
            
            # Загружаем checkpoint с моделью, импутером и скейлером
            # Отключаем weights_only для совместимости с sklearn объектами
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                # Fallback для старых версий PyTorch
                self.logger.warning(f"Fallback к старому методу загрузки: {str(e)}")
                checkpoint = torch.load(model_path, map_location=self.device)
            
            # Извлекаем компоненты
            model_state_dict = checkpoint['model_state_dict']
            self.imputer = checkpoint['imputer']
            self.scaler = checkpoint['scaler']
            
            # Получаем параметры модели из метаданных
            input_dim = self.model_metadata['input_dim']
            hidden_dims = self.model_metadata['hidden_dims']
            self.feature_names = self.model_metadata['feature_names']
            self.model_type = model_type.lower()
            
            # Создаем модель нужного типа
            if self.model_type == 'attention':
                attention_dim = self.model_metadata.get('attention_dim', 64)
                self.model = self.DVMHAttentionModel(input_dim, hidden_dims, attention_dim)
            else:
                self.model = self.DVMHEfficiencyMLP(input_dim, hidden_dims)
            
            # Загружаем веса
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Модель {model_name} загружена успешно")
            self.logger.info(f"Размерность входа: {input_dim}")
            self.logger.info(f"Архитектура: {hidden_dims}")
            self.logger.info(f"Количество признаков: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            return False
    
    def process_fortran_file(self, fortran_file_path: str) -> Optional[Dict]:
        """
        Обработка Fortran файла для получения статистики покрытия
        Использует существующий метод из DVMHDataCollector
        
        Args:
            fortran_file_path: Путь к Fortran файлу
            
        Returns:
            Dict или None: Статистика покрытия в формате info.json
        """
        self.logger.info(f"Обработка Fortran файла: {fortran_file_path}")
        
        if not os.path.exists(fortran_file_path):
            self.logger.error(f"Fortran файл не найден: {fortran_file_path}")
            return None
        
        # Создаем временную директорию
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_source_dir = os.path.join(temp_dir, "sources")
            temp_results_dir = os.path.join(temp_dir, "results")
            
            os.makedirs(temp_source_dir, exist_ok=True)
            os.makedirs(temp_results_dir, exist_ok=True)
            
            try:
                # Копируем файл во временную директорию
                filename = os.path.basename(fortran_file_path)
                temp_file_path = os.path.join(temp_source_dir, filename)
                shutil.copy2(fortran_file_path, temp_file_path)
                
                # Используем существующий метод из DVMHDataCollector
                success = self.data_collector._process_single_file(
                    temp_source_dir, 
                    temp_results_dir,
                    filename
                )
                
                if not success:
                    self.logger.error(f"Ошибка при обработке файла {fortran_file_path}")
                    return None
                
                # Загружаем результат
                file_name = filename.split('.')[0]
                info_json_path = os.path.join(temp_results_dir, file_name, "info.json")
                
                if not os.path.exists(info_json_path):
                    self.logger.error(f"Файл info.json не был создан: {info_json_path}")
                    return None
                
                with open(info_json_path, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                
                self.logger.info(f"Fortran файл обработан успешно")
                return info_data
                
            except Exception as e:
                self.logger.error(f"Ошибка при обработке Fortran файла: {str(e)}")
                return None
    
    def extract_features_from_info(self, info_data: Dict, grid: List[int], 
                                  threads: int, parallel_time: float) -> Optional[pd.DataFrame]:
        """
        Извлечение признаков из данных info.json
        Использует существующие методы из DVMHFeatureSpaceCreator
        
        Args:
            info_data: Данные из info.json
            grid: Сетка процессоров
            threads: Количество нитей
            parallel_time: Время параллельного выполнения
            
        Returns:
            pd.DataFrame или None: Признаки для предсказания
        """
        self.logger.info("Извлечение признаков из статистики покрытия")
        
        try:
            # Проверяем формат данных
            if not isinstance(info_data, list) or len(info_data) < 2:
                self.logger.error("Некорректный формат данных info.json")
                return None
            
            program_info = info_data[1].get('program_info', {})
            
            # Добавляем информацию о запуске для совместимости
            program_info['launches'] = [{
                'grid': grid,
                'threads': threads,
                'total_time': parallel_time
            }]
            
            # Используем существующие методы из DVMHFeatureSpaceCreator
            static_features = self.feature_extractor.extract_static_program_features(program_info)
            launch_features = self.feature_extractor.extract_launch_configuration_features(grid, threads)
            
            # Объединяем признаки
            all_features = {**static_features, **launch_features}
            
            # Добавляем время параллельного выполнения
            all_features['parallel_execution_time'] = parallel_time
            
            # Создаем DataFrame
            features_df = pd.DataFrame([all_features])
            
            self.logger.info(f"Извлечено признаков: {len(all_features)}")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении признаков: {str(e)}")
            return None
    
    def preprocess_features(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Предобработка признаков с консистентным кодированием (ИСПРАВЛЕНО)
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            np.ndarray или None: Обработанные признаки
        """
        self.logger.info("Предобработка признаков (исправленная версия)")
        
        if self.imputer is None or self.scaler is None:
            self.logger.error("Препроцессоры не загружены. Сначала загрузите модель.")
            return None
        
        try:
            features_df = features_df.copy()
            
            # Убираем служебные колонки
            exclude_columns = ['parallel_execution_time', 'launch_config', 
                             'normalized_parallel_time', 'efficiency', 
                             'program_name', 'target_speedup']
            
            columns_to_drop = [col for col in exclude_columns if col in features_df.columns]
            if columns_to_drop:
                self.logger.debug(f"Удаляем колонки: {columns_to_drop}")
                features_df = features_df.drop(columns=columns_to_drop)
            
            # ИСПРАВЛЕНИЕ: Обрабатываем категориальные признаки как при обучении
            categorical_columns = features_df.select_dtypes(include=['object']).columns.tolist()
            if 'program_name' in categorical_columns:
                categorical_columns.remove('program_name')
            
            for col in categorical_columns:
                # Заполняем NaN как при обучении
                features_df[col] = features_df[col].fillna("unknown")
                
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Используем детерминистичное кодирование
                if features_df[col].dtype == 'object':
                    # Простое детерминистичное кодирование вместо hash
                    unique_values = sorted(features_df[col].unique())
                    category_mapping = {val: idx for idx, val in enumerate(unique_values)}
                    features_df[col] = features_df[col].map(category_mapping)
                    self.logger.debug(f"Категории для {col}: {category_mapping}")
            
            # Обрабатываем смешанные типы как при обучении
            for col in features_df.columns:
                if features_df[col].dtype == 'object':
                    try:
                        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                    except:
                        features_df[col] = features_df[col].fillna("unknown").astype('category').cat.codes
            
            # Убеждаемся в правильном порядке признаков
            if self.feature_names:
                missing_features = set(self.feature_names) - set(features_df.columns)
                extra_features = set(features_df.columns) - set(self.feature_names)
                
                if missing_features:
                    self.logger.warning(f"Отсутствующие признаки (заполняем нулями): {missing_features}")
                    for feature in missing_features:
                        features_df[feature] = 0
                
                if extra_features:
                    self.logger.debug(f"Лишние признаки (удаляем): {extra_features}")
                    features_df = features_df.drop(columns=list(extra_features))
                
                # КРИТИЧЕСКИ ВАЖНО: Строгий порядок как при обучении
                features_df = features_df[self.feature_names]
                
                self.logger.info(f"Финальная размерность признаков: {features_df.shape}")
                self.logger.debug(f"Первые 5 признаков: {list(features_df.columns[:5])}")
            
            # Применяем препроцессоры
            features_imputed = self.imputer.transform(features_df)
            features_scaled = self.scaler.transform(features_imputed)
            
            features_array = features_scaled.astype(np.float32)
            
            # Финальная проверка и отладка
            nan_count = np.isnan(features_array).sum()
            if nan_count > 0:
                self.logger.warning(f"Найдено NaN значений: {nan_count}. Заменяем на 0.")
                features_array = np.nan_to_num(features_array, nan=0.0)
            
            # Отладочная информация
            self.logger.info(f"Предобработка завершена. Размерность: {features_array.shape}")
            self.logger.debug(f"Статистика: min={features_array.min():.4f}, max={features_array.max():.4f}, mean={features_array.mean():.4f}")
            
            # Проверка на выбросы
            outliers = np.abs(features_array) > 5
            if outliers.any():
                outlier_count = outliers.sum()
                self.logger.warning(f"Обнаружены выбросы (|x| > 5): {outlier_count}")
            
            return features_array
            
        except Exception as e:
            self.logger.error(f"Ошибка при предобработке признаков: {str(e)}")
            return None
    
    def predict_speedup(self, features: np.ndarray) -> Optional[Union[float, Tuple[float, np.ndarray]]]:
        """
        Предсказание ускорения
        
        Args:
            features: Обработанные признаки
            
        Returns:
            float или Tuple: Предсказанное ускорение (и веса внимания для attention модели)
        """
        if self.model is None:
            self.logger.error("Модель не загружена")
            return None
        
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Убеждаемся, что features имеет правильную размерность
                if len(features.shape) == 1:
                    # Добавляем batch dimension если его нет
                    features = features.reshape(1, -1)
                
                self.logger.debug(f"Размерность входных признаков: {features.shape}")
                
                # Преобразуем в тензор
                features_tensor = torch.FloatTensor(features).to(self.device)
                
                # Делаем предсказание
                if self.model_type == 'attention':
                    prediction, attention_weights = self.model(features_tensor)
                    
                    # Извлекаем скалярное значение
                    if prediction.dim() > 0:
                        speedup = float(prediction.cpu().numpy()[0])
                    else:
                        speedup = float(prediction.cpu().numpy())
                    
                    attention_weights = attention_weights.cpu().numpy()
                    
                    self.logger.info(f"Предсказанное ускорение: {speedup:.4f}")
                    return speedup, attention_weights
                else:
                    prediction = self.model(features_tensor)
                    
                    # Извлекаем скалярное значение
                    if prediction.dim() > 0:
                        speedup = float(prediction.cpu().numpy()[0])
                    else:
                        speedup = float(prediction.cpu().numpy())
                    
                    self.logger.info(f"Предсказанное ускорение: {speedup:.4f}")
                    return speedup
                
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании: {str(e)}")
            self.logger.debug(f"Размерность features: {features.shape if hasattr(features, 'shape') else 'N/A'}")
            if hasattr(features, 'shape'):
                self.logger.debug(f"Содержимое features (первые 10): {features.flatten()[:10]}")
            return None
    
    def predict_from_fortran(self, fortran_file_path: str, grid: List[int], 
                           threads: int, parallel_time: float) -> Optional[Union[float, Tuple[float, np.ndarray]]]:
        """
        Полный цикл предсказания из Fortran файла
        
        Args:
            fortran_file_path: Путь к Fortran файлу
            grid: Сетка процессоров
            threads: Количество нитей
            parallel_time: Время параллельного выполнения
            
        Returns:
            float или Tuple: Предсказанное ускорение (и веса внимания для attention модели)
        """
        self.logger.info(f"Предсказание из Fortran файла: {fortran_file_path}")
        
        # 1. Обрабатываем Fortran файл
        info_data = self.process_fortran_file(fortran_file_path)
        if info_data is None:
            return None
        
        # 2. Извлекаем признаки
        features_df = self.extract_features_from_info(info_data, grid, threads, parallel_time)
        if features_df is None:
            return None
        
        # 3. Предобрабатываем признаки
        features_array = self.preprocess_features(features_df)
        if features_array is None:
            return None
        
        # 4. Делаем предсказание
        result = self.predict_speedup(features_array)
        return result
    
    def predict_from_json(self, json_file_path: str, grid: List[int], 
                         threads: int, parallel_time: float) -> Optional[Union[float, Tuple[float, np.ndarray]]]:
        """
        Полный цикл предсказания из JSON файла
        
        Args:
            json_file_path: Путь к JSON файлу со статистикой
            grid: Сетка процессоров
            threads: Количество нитей
            parallel_time: Время параллельного выполнения
            
        Returns:
            float или Tuple: Предсказанное ускорение (и веса внимания для attention модели)
        """
        self.logger.info(f"Предсказание из JSON файла: {json_file_path}")
        
        # 1. Загружаем JSON данные
        if not os.path.exists(json_file_path):
            self.logger.error(f"JSON файл не найден: {json_file_path}")
            return None
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке JSON: {str(e)}")
            return None
        
        # 2-4. Используем тот же пайплайн что и для Fortran файлов
        features_df = self.extract_features_from_info(info_data, grid, threads, parallel_time)
        if features_df is None:
            return None
        
        features_array = self.preprocess_features(features_df)
        if features_array is None:
            return None
        
        result = self.predict_speedup(features_array)
        return result
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Получение информации о загруженной модели"""
        if self.model_metadata is None:
            return None
        
        return {
            'model_name': model_name,
            'model_type': self.model_metadata.get('model_type', 'Unknown'),
            'input_dim': self.model_metadata.get('input_dim', 0),
            'hidden_dims': self.model_metadata.get('hidden_dims', []),
            'metrics': self.model_metadata.get('metrics', {}),
            'test_programs': self.model_metadata.get('test_programs', []),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'device': str(self.device)
        }
    
    def list_available_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.pt'):
                model_name = file[:-3]  # Убираем расширение .pt
                models.append(model_name)
        
        return models