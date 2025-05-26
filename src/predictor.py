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
import re

from .feature_preprocessor import DVMHFeaturePreprocessor

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DVMHPerformancePredictor:
    """Класс для предсказания эффективности DVMH программ"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Инициализация предиктора (ОБНОВЛЕННАЯ ВЕРСИЯ)
        
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
        self.preprocessor = None  # Заменяем scaler, imputer на preprocessor
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
    
    # def load_model(self, model_name: str, model_type: str = 'mlp') -> bool:
    #     """
    #     Загрузка обученной модели с препроцессором (ОБНОВЛЕННАЯ ВЕРСИЯ)
        
    #     Args:
    #         model_name: Имя модели (без расширения .pt)
    #         model_type: Тип модели ('mlp' или 'attention')
            
    #     Returns:
    #         bool: True если модель загружена успешно
    #     """
    #     self.logger.info(f"Загрузка модели: {model_name} (тип: {model_type})")
        
    #     # Пути к файлам модели
    #     model_path = os.path.join(self.models_dir, f"{model_name}.pt")
    #     metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
    #     # Проверяем существование файлов
    #     if not os.path.exists(model_path):
    #         self.logger.error(f"Файл модели не найден: {model_path}")
    #         return False
        
    #     if not os.path.exists(metadata_path):
    #         self.logger.error(f"Файл метаданных не найден: {metadata_path}")
    #         return False
        
    #     try:
    #         # Загружаем метаданные
    #         with open(metadata_path, 'r', encoding='utf-8') as f:
    #             self.model_metadata = json.load(f)
            
    #         # Загружаем checkpoint с моделью и препроцессором
    #         try:
    #             checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
    #         except Exception as e:
    #             # Fallback для старых версий PyTorch
    #             self.logger.warning(f"Fallback к старому методу загрузки: {str(e)}")
    #             checkpoint = torch.load(model_path, map_location=self.device)
            
    #         # Извлекаем компоненты
    #         model_state_dict = checkpoint['model_state_dict']
            
    #         # ГЛАВНОЕ ИЗМЕНЕНИЕ: загружаем препроцессор вместо отдельных компонентов
    #         if 'preprocessor' in checkpoint:
    #             self.preprocessor = checkpoint['preprocessor']
    #             self.logger.info("Загружен новый препроцессор")
    #         else:
    #             # Обратная совместимость со старыми моделями
    #             self.logger.warning("Старый формат модели, создаем fallback препроцессор")
    #             self.preprocessor = self._create_fallback_preprocessor(checkpoint)
            
    #         # Получаем параметры модели из метаданных
    #         input_dim = self.model_metadata['input_dim']
    #         hidden_dims = self.model_metadata['hidden_dims']
    #         self.feature_names = self.model_metadata.get('feature_names', [])
    #         self.model_type = model_type.lower()
            
    #         # Создаем модель нужного типа
    #         if self.model_type == 'attention':
    #             attention_dim = self.model_metadata.get('attention_dim', 64)
    #             self.model = self.DVMHAttentionModel(input_dim, hidden_dims, attention_dim)
    #         else:
    #             self.model = self.DVMHEfficiencyMLP(input_dim, hidden_dims)
            
    #         # Загружаем веса
    #         self.model.load_state_dict(model_state_dict)
    #         self.model.to(self.device)
    #         self.model.eval()
            
    #         self.logger.info(f"Модель {model_name} загружена успешно")
    #         self.logger.info(f"Размерность входа: {input_dim}")
    #         self.logger.info(f"Архитектура: {hidden_dims}")
    #         self.logger.info(f"Количество признаков: {len(self.feature_names)}")
            
    #         return True
            
    #     except Exception as e:
    #         self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
    #         return False
        
    def _create_fallback_preprocessor(self, checkpoint: Dict) -> DVMHFeaturePreprocessor:
        """
        Создание fallback препроцессора для обратной совместимости
        
        Args:
            checkpoint: Данные из старой модели
            
        Returns:
            DVMHFeaturePreprocessor: Простой препроцессор
        """
        self.logger.warning("Создание fallback препроцессора для старой модели")
        
        preprocessor = DVMHFeaturePreprocessor()
        
        # Переносим старые компоненты если есть
        if 'imputer' in checkpoint:
            preprocessor.imputer = checkpoint['imputer']
        if 'scaler' in checkpoint:
            preprocessor.scaler = checkpoint['scaler']
        
        # Устанавливаем базовые параметры
        preprocessor.feature_order = self.feature_names
        preprocessor.is_fitted = True
        
        return preprocessor
    
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
        Предобработка признаков с использованием обученного препроцессора (НОВАЯ ВЕРСИЯ)
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            np.ndarray или None: Обработанные признаки
        """
        self.logger.info("Предобработка признаков (новая версия с препроцессором)")
        
        if self.preprocessor is None:
            self.logger.error("Препроцессор не загружен. Сначала загрузите модель.")
            return None
        
        try:
            # Используем препроцессор для обработки
            features_array = self.preprocessor.transform(features_df)
            
            self.logger.info(f"Предобработка завершена. Размерность: {features_array.shape}")
            self.logger.debug(f"Статистика: min={features_array.min():.4f}, max={features_array.max():.4f}, mean={features_array.mean():.4f}")
            
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
    
    def list_available_models(self) -> List[str]:
        """Получение списка доступных моделей (включая дообученные)"""
        models = []
        
        # Обычные модели
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('.pt'):
                    model_name = file[:-3]
                    models.append(model_name)
        
        # Дообученные модели
        finetune_dir = os.path.join(self.models_dir, 'finetuned')
        if os.path.exists(finetune_dir):
            for file in os.listdir(finetune_dir):
                if file.endswith('.pt'):
                    model_name = file[:-3]
                    models.append(f"finetuned/{model_name}")
        
        return sorted(models)

    def load_model(self, model_name: str, model_type: str = 'mlp') -> bool:
        """
        Загрузка модели с поддержкой дообученных моделей (ОБНОВЛЕННАЯ ВЕРСИЯ)
        
        Args:
            model_name: Имя модели (может включать префикс finetuned/)
            model_type: Тип модели ('mlp' или 'attention')
            
        Returns:
            bool: True если модель загружена успешно
        """
        self.logger.info(f"Загрузка модели: {model_name} (тип: {model_type})")
        
        # Определяем пути к файлам
        if model_name.startswith('finetuned/'):
            # Дообученная модель
            actual_model_name = model_name[10:]  # Убираем префикс 'finetuned/'
            model_path = os.path.join(self.models_dir, 'finetuned', f"{actual_model_name}.pt")
            metadata_path = os.path.join(self.models_dir, 'finetuned', f"{actual_model_name}_metadata.json")
            self.logger.info(f"Загрузка дообученной модели: {actual_model_name}")
        else:
            # Обычная модель
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
            
            # Определяем тип модели автоматически, если это дообученная модель
            if 'model_type' in self.model_metadata:
                detected_type = self.model_metadata['model_type'].lower()
                if 'attention' in detected_type:
                    model_type = 'attention'
                else:
                    model_type = 'mlp'
                self.logger.info(f"Автоматически определен тип модели: {model_type}")
            
            # Загружаем checkpoint
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                self.logger.warning(f"Fallback к старому методу загрузки: {str(e)}")
                checkpoint = torch.load(model_path, map_location=self.device)
            
            # Извлекаем компоненты
            model_state_dict = checkpoint['model_state_dict']
            
            # Загружаем препроцессор
            if 'preprocessor' in checkpoint:
                self.preprocessor = checkpoint['preprocessor']
                self.logger.info("Загружен препроцессор")
            else:
                self.logger.warning("Старый формат модели, создаем fallback препроцессор")
                self.preprocessor = self._create_fallback_preprocessor(checkpoint)
            
            # Получаем параметры модели
            input_dim = self.model_metadata['input_dim']
            hidden_dims = self.model_metadata['hidden_dims']
            self.feature_names = self.model_metadata.get('feature_names', [])
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
            
            # Дополнительная информация для дообученных моделей
            if model_name.startswith('finetuned/'):
                if 'base_model_name' in self.model_metadata:
                    self.logger.info(f"Базовая модель: {self.model_metadata['base_model_name']}")
                if 'fine_tune_strategy' in self.model_metadata:
                    self.logger.info(f"Стратегия дообучения: {self.model_metadata['fine_tune_strategy']}")
                if 'fine_tune_timestamp' in self.model_metadata:
                    self.logger.info(f"Дата дообучения: {self.model_metadata['fine_tune_timestamp']}")
            
            self.logger.info(f"Модель {model_name} загружена успешно")
            self.logger.info(f"Размерность входа: {input_dim}")
            self.logger.info(f"Архитектура: {hidden_dims}")
            self.logger.info(f"Количество признаков: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели: {str(e)}")
            return False

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Получение информации о загруженной модели (ОБНОВЛЕННАЯ ВЕРСИЯ)"""
        if self.model_metadata is None:
            return None
        
        info = {
            'model_name': model_name,
            'model_type': self.model_metadata.get('model_type', 'Unknown'),
            'input_dim': self.model_metadata.get('input_dim', 0),
            'hidden_dims': self.model_metadata.get('hidden_dims', []),
            'metrics': self.model_metadata.get('metrics', {}),
            'test_programs': self.model_metadata.get('test_programs', []),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'device': str(self.device),
            'is_finetuned': self.model_metadata.get('is_finetuned', False)
        }
        
        # Дополнительная информация для дообученных моделей
        if info['is_finetuned']:
            info.update({
                'base_model_name': self.model_metadata.get('base_model_name', 'Unknown'),
                'fine_tune_strategy': self.model_metadata.get('fine_tune_strategy', 'Unknown'),
                'fine_tune_timestamp': self.model_metadata.get('fine_tune_timestamp', 'Unknown'),
                'adaptation_strategy': self.model_metadata.get('adaptation_strategy', 'Unknown')
            })
        
        return info

    def list_finetuned_models(self) -> List[Dict]:
        """Получение списка всех дообученных моделей с подробной информацией"""
        finetune_dir = os.path.join(self.models_dir, 'finetuned')
        
        if not os.path.exists(finetune_dir):
            return []
        
        models = []
        for file in os.listdir(finetune_dir):
            if file.endswith('.pt'):
                model_name = file[:-3]
                metadata_path = os.path.join(finetune_dir, f"{model_name}_metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        models.append({
                            'name': f"finetuned/{model_name}",
                            'short_name': model_name,
                            'base_model': metadata.get('base_model_name', 'Unknown'),
                            'strategy': metadata.get('fine_tune_strategy', 'Unknown'),
                            'timestamp': metadata.get('fine_tune_timestamp', 'Unknown'),
                            'model_type': metadata.get('model_type', 'Unknown'),
                            'r2_score': metadata.get('metrics', {}).get('r2', 'Unknown'),
                            'path': os.path.join(finetune_dir, file),
                            'is_finetuned': True
                        })
                    except:
                        continue
        
        # Сортируем по времени создания
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models

    def compare_model_performance(self, model_names: List[str], 
                                test_data_path: str = None) -> Optional[Dict]:
        """
        Сравнение производительности нескольких моделей
        
        Args:
            model_names: Список имен моделей для сравнения
            test_data_path: Путь к тестовым данным (опционально)
            
        Returns:
            Dict: Результаты сравнения моделей
        """
        if not test_data_path or not os.path.exists(test_data_path):
            self.logger.warning("Тестовые данные не предоставлены для сравнения")
            return None
        
        try:
            import pandas as pd
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Загружаем тестовые данные
            test_df = pd.read_csv(test_data_path, low_memory=False)
            self.logger.info(f"Загружены тестовые данные: {test_df.shape}")
            
            if 'target_speedup' not in test_df.columns:
                self.logger.error("Колонка 'target_speedup' не найдена в тестовых данных")
                return None
            
            results = {}
            current_model_backup = None
            current_preprocessor_backup = None
            
            # Сохраняем текущую модель
            if self.model is not None:
                current_model_backup = self.model
                current_preprocessor_backup = self.preprocessor
            
            for model_name in model_names:
                self.logger.info(f"Тестирование модели: {model_name}")
                
                try:
                    # Определяем тип модели
                    model_type = 'attention' if 'attention' in model_name.lower() else 'mlp'
                    
                    # Загружаем модель
                    if not self.load_model(model_name, model_type):
                        self.logger.error(f"Не удалось загрузить модель {model_name}")
                        continue
                    
                    # Получаем предсказания для всех строк
                    predictions = []
                    actual_values = []
                    
                    for idx, row in test_df.iterrows():
                        try:
                            # Извлекаем параметры запуска
                            if 'launch_config' in row and pd.notna(row['launch_config']):
                                # Парсим launch_config
                                config_str = str(row['launch_config'])
                                grid_match = re.search(r'grid=\[([^\]]+)\]', config_str)
                                threads_match = re.search(r'threads=(\d+)', config_str)
                                
                                if grid_match and threads_match:
                                    grid = [int(x.strip()) for x in grid_match.group(1).split(',')]
                                    threads = int(threads_match.group(1))
                                else:
                                    # Значения по умолчанию
                                    grid = [1, 1]
                                    threads = 1
                            else:
                                # Используем значения из других колонок если есть
                                grid = [int(row.get('launch_grid_1', 1)), int(row.get('launch_grid_2', 1))]
                                threads = int(row.get('launch_threads', 1))
                            
                            parallel_time = row.get('parallel_execution_time', 1.0)
                            actual_speedup = row['target_speedup']
                            
                            # Создаем временный DataFrame для предсказания
                            temp_df = pd.DataFrame([row])
                            
                            # Получаем предсказание
                            features_array = self.preprocess_features(temp_df)
                            if features_array is not None:
                                result = self.predict_speedup(features_array)
                                
                                if isinstance(result, tuple):  # Attention модель
                                    predicted_speedup = result[0]
                                else:
                                    predicted_speedup = result
                                
                                predictions.append(predicted_speedup)
                                actual_values.append(actual_speedup)
                        
                        except Exception as e:
                            self.logger.debug(f"Ошибка при обработке строки {idx}: {str(e)}")
                            continue
                    
                    # Вычисляем метрики
                    if predictions and actual_values:
                        mse = mean_squared_error(actual_values, predictions)
                        mae = mean_absolute_error(actual_values, predictions)
                        r2 = r2_score(actual_values, predictions)
                        rmse = np.sqrt(mse)
                        
                        results[model_name] = {
                            'mse': mse,
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2,
                            'predictions_count': len(predictions),
                            'model_info': self.get_model_info(model_name)
                        }
                        
                        self.logger.info(f"Модель {model_name} - R²: {r2:.4f}, MAE: {mae:.4f}")
                    else:
                        self.logger.warning(f"Не удалось получить предсказания для модели {model_name}")
                
                except Exception as e:
                    self.logger.error(f"Ошибка при тестировании модели {model_name}: {str(e)}")
                    continue
            
            # Восстанавливаем исходную модель
            if current_model_backup is not None:
                self.model = current_model_backup
                self.preprocessor = current_preprocessor_backup
            
            if results:
                # Определяем лучшую модель
                best_model = max(results.keys(), key=lambda k: results[k]['r2'])
                
                comparison_results = {
                    'models_tested': len(results),
                    'results': results,
                    'best_model': best_model,
                    'best_r2': results[best_model]['r2'],
                    'ranking': sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)
                }
                
                self.logger.info(f"Сравнение завершено. Лучшая модель: {best_model} (R² = {results[best_model]['r2']:.4f})")
                return comparison_results
            else:
                self.logger.error("Не удалось протестировать ни одну модель")
                return None
        
        except Exception as e:
            self.logger.error(f"Ошибка при сравнении моделей: {str(e)}")
            return None