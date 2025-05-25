import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import logging
import os
import yaml
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import joblib

from .model_trainer import DVMHEfficiencyMLP, DVMHAttentionModel
from .feature_extractor import DVMHFeatureSpaceCreator

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
        prediction_config = self.config.get('prediction', {})
        
        self.models_dir = model_config.get('models_directory', './models')
        self.predictions_dir = prediction_config.get('output_directory', './predictions')
        
        # Создаем директории
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Настройка устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загруженные модели и метаданные
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        
        self.logger.info("DVMHPerformancePredictor инициализирован")
    
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
        Загрузка обученной модели
        
        Args:
            model_name: Имя модели (без расширения)
            model_type: Тип модели ('mlp' или 'attention')
            
        Returns:
            bool: True если модель загружена успешно
        """
        try:
            # Пути к файлам
            model_path = os.path.join(self.models_dir, f"{model_name}.pt")
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            
            if not os.path.exists(model_path):
                self.logger.error(f"Файл модели не найден: {model_path}")
                return False
            
            if not os.path.exists(metadata_path):
                self.logger.error(f"Файл метаданных не найден: {metadata_path}")
                return False
            
            # Загружаем метаданные
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Создаем модель
            input_dim = metadata['input_dim']
            hidden_dims = metadata['hidden_dims']
            
            if model_type.lower() == 'mlp':
                model = DVMHEfficiencyMLP(input_dim, hidden_dims)
            elif model_type.lower() == 'attention':
                attention_dim = metadata.get('attention_dim', 64)
                model = DVMHAttentionModel(input_dim, hidden_dims, attention_dim)
            else:
                self.logger.error(f"Неизвестный тип модели: {model_type}")
                return False
            
            # Загружаем веса
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # Сохраняем модель и метаданные
            self.models[model_name] = {
                'model': model,
                'metadata': metadata,
                'type': model_type
            }
            
            self.logger.info(f"Модель {model_name} загружена успешно")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке модели {model_name}: {str(e)}")
            return False
    
    def load_scaler(self, scaler_path: str) -> bool:
        """
        Загрузка скейлера для нормализации данных
        
        Args:
            scaler_path: Путь к файлу скейлера
            
        Returns:
            bool: True если скейлер загружен успешно
        """
        try:
            if not os.path.exists(scaler_path):
                self.logger.error(f"Файл скейлера не найден: {scaler_path}")
                return False
            
            scaler = joblib.load(scaler_path)
            self.scalers['default'] = scaler
            
            self.logger.info(f"Скейлер загружен из: {scaler_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке скейлера: {str(e)}")
            return False
    
    def predict_single_configuration(self, program_data: Dict, grid: List[int], 
                                   threads: int, model_name: str = None, 
                                   dvmh_processors: int = 1) -> Dict:
        """
        Предсказание эффективности для одной конфигурации запуска
        
        Args:
            program_data: Данные программы (результат агрегации)
            grid: Размерности сетки процессоров
            threads: Количество нитей
            model_name: Имя модели для предсказания
            dvmh_processors: Количество DVMH процессоров
            
        Returns:
            Dict: Результаты предсказания
        """
        try:
            # Выбираем модель
            if model_name is None:
                if not self.models:
                    raise ValueError("Не загружено ни одной модели")
                model_name = list(self.models.keys())[0]
            
            if model_name not in self.models:
                raise ValueError(f"Модель {model_name} не загружена")
            
            model_info = self.models[model_name]
            model = model_info['model']
            metadata = model_info['metadata']
            
            # Создаем экстрактор признаков
            feature_extractor = DVMHFeatureSpaceCreator(dvmh_processors=dvmh_processors)
            
            # Извлекаем статические признаки программы
            program_info = program_data.get('program_info', {})
            static_features = feature_extractor.extract_static_program_features(program_info)
            
            # Извлекаем признаки конфигурации запуска
            launch_features = feature_extractor.extract_launch_configuration_features(grid, threads)
            
            # Объединяем все признаки
            all_features = {**static_features, **launch_features}
            
            # Создаем DataFrame
            feature_names = metadata['feature_names']
            feature_vector = []
            
            for feature_name in feature_names:
                if feature_name in all_features:
                    feature_vector.append(all_features[feature_name])
                else:
                    self.logger.warning(f"Признак {feature_name} не найден, используем 0")
                    feature_vector.append(0.0)
            
            # Преобразуем в numpy array
            X = np.array(feature_vector).reshape(1, -1).astype(np.float32)
            
            # Нормализуем данные
            if 'default' in self.scalers:
                X = self.scalers['default'].transform(X)
            else:
                self.logger.warning("Скейлер не загружен, используем данные без нормализации")
            
            # Обрабатываем NaN и бесконечные значения
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Делаем предсказание
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                
                if model_info['type'] == 'attention':
                    prediction, attention_weights = model(X_tensor)
                    attention_weights = attention_weights.cpu().numpy().flatten()
                else:
                    prediction = model(X_tensor)
                    attention_weights = None
                
                predicted_speedup = prediction.cpu().numpy()[0]
            
            # Формируем результат
            result = {
                'predicted_speedup': float(predicted_speedup),
                'grid': grid,
                'threads': threads,
                'total_processors': np.prod(grid) * threads,
                'model_used': model_name,
                'model_type': model_info['type'],
                'dvmh_processors': dvmh_processors,
                'features_used': len(feature_names)
            }
            
            if attention_weights is not None:
                # Находим топ-5 наиболее важных признаков
                top_indices = np.argsort(attention_weights)[-5:][::-1]
                result['top_important_features'] = [
                    {
                        'feature': feature_names[i],
                        'attention_weight': float(attention_weights[i]),
                        'feature_value': float(feature_vector[i])
                    }
                    for i in top_indices
                ]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при предсказании: {str(e)}")
            return {
                'error': str(e),
                'predicted_speedup': 0.0,
                'grid': grid,
                'threads': threads
            }
    
    def predict_multiple_configurations(self, program_data: Dict, 
                                       configurations: List[Dict],
                                       model_name: str = None,
                                       dvmh_processors: int = 1) -> List[Dict]:
        """
        Предсказание эффективности для множества конфигураций
        
        Args:
            program_data: Данные программы
            configurations: Список конфигураций [{'grid': [2,2], 'threads': 4}, ...]
            model_name: Имя модели
            dvmh_processors: Количество DVMH процессоров
            
        Returns:
            List[Dict]: Список результатов предсказаний
        """
        results = []
        
        for i, config in enumerate(configurations):
            self.logger.debug(f"Предсказание для конфигурации {i+1}/{len(configurations)}")
            
            result = self.predict_single_configuration(
                program_data=program_data,
                grid=config['grid'],
                threads=config['threads'],
                model_name=model_name,
                dvmh_processors=dvmh_processors
            )
            
            results.append(result)
        
        return results
    
    def find_optimal_configuration(self, program_data: Dict, 
                                  max_processors: int = 16,
                                  model_name: str = None,
                                  dvmh_processors: int = 1) -> Dict:
        """
        Поиск оптимальной конфигурации запуска
        
        Args:
            program_data: Данные программы
            max_processors: Максимальное количество процессоров
            model_name: Имя модели
            dvmh_processors: Количество DVMH процессоров
            
        Returns:
            Dict: Оптимальная конфигурация и результаты
        """
        self.logger.info("Поиск оптимальной конфигурации...")
        
        # Генерируем возможные конфигурации
        configurations = self._generate_configurations(max_processors)
        
        # Получаем предсказания для всех конфигураций
        results = self.predict_multiple_configurations(
            program_data, configurations, model_name, dvmh_processors
        )
        
        # Находим конфигурацию с максимальным ускорением
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {
                'error': 'Не удалось получить валидные предсказания',
                'configurations_tested': len(configurations)
            }
        
        best_result = max(valid_results, key=lambda x: x['predicted_speedup'])
        
        # Дополнительная информация
        optimization_info = {
            'best_configuration': best_result,
            'configurations_tested': len(configurations),
            'valid_predictions': len(valid_results),
            'speedup_range': {
                'min': min(r['predicted_speedup'] for r in valid_results),
                'max': max(r['predicted_speedup'] for r in valid_results),
                'mean': np.mean([r['predicted_speedup'] for r in valid_results])
            },
            'all_results': valid_results
        }
        
        self.logger.info(f"Оптимальная конфигурация найдена: grid={best_result['grid']}, "
                        f"threads={best_result['threads']}, speedup={best_result['predicted_speedup']:.2f}")
        
        return optimization_info
    
    def _generate_configurations(self, max_processors: int) -> List[Dict]:
        """Генерация возможных конфигураций запуска"""
        configurations = []
        
        # Генерируем различные комбинации сеток и потоков
        for total_procs in range(1, max_processors + 1):
            for threads in [1, 2, 4, 8, 16]:
                if threads > total_procs:
                    continue
                
                grid_size = total_procs // threads
                
                # 1D сетка
                configurations.append({
                    'grid': [grid_size],
                    'threads': threads
                })
                
                # 2D сетки
                for i in range(1, int(np.sqrt(grid_size)) + 1):
                    if grid_size % i == 0:
                        j = grid_size // i
                        configurations.append({
                            'grid': [i, j],
                            'threads': threads
                        })
                
                # 3D сетки (для больших размеров)
                if grid_size >= 8:
                    for i in range(1, int(grid_size**(1/3)) + 1):
                        if grid_size % i == 0:
                            remaining = grid_size // i
                            for j in range(1, int(np.sqrt(remaining)) + 1):
                                if remaining % j == 0:
                                    k = remaining // j
                                    configurations.append({
                                        'grid': [i, j, k],
                                        'threads': threads
                                    })
        
        # Удаляем дубликаты
        unique_configs = []
        seen = set()
        
        for config in configurations:
            key = (tuple(sorted(config['grid'])), config['threads'])
            if key not in seen:
                seen.add(key)
                unique_configs.append(config)
        
        self.logger.debug(f"Сгенерировано {len(unique_configs)} уникальных конфигураций")
        return unique_configs
    
    def save_predictions(self, predictions: Any, filename: str, 
                        format: str = 'json') -> str:
        """
        Сохранение результатов предсказаний
        
        Args:
            predictions: Результаты предсказаний
            filename: Имя файла
            format: Формат сохранения ('json' или 'csv')
            
        Returns:
            str: Путь к сохраненному файлу
        """
        output_path = os.path.join(self.predictions_dir, filename)
        
        try:
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                if isinstance(predictions, list):
                    df = pd.DataFrame(predictions)
                    df.to_csv(output_path, index=False)
                else:
                    raise ValueError("Для CSV формата predictions должен быть списком")
            
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")
            
            self.logger.info(f"Предсказания сохранены в: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении предсказаний: {str(e)}")
            raise
    
    def load_program_data(self, program_file: str) -> Dict:
        """
        Загрузка данных программы из файла
        
        Args:
            program_file: Путь к файлу с данными программы
            
        Returns:
            Dict: Данные программы
        """
        try:
            with open(program_file, 'r', encoding='utf-8') as f:
                program_data = json.load(f)
            
            self.logger.info(f"Данные программы загружены из: {program_file}")
            return program_data
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных программы: {str(e)}")
            raise
    
    def batch_predict_from_file(self, input_file: str, output_file: str = None,
                               model_name: str = None) -> str:
        """
        Пакетное предсказание из файла
        
        Args:
            input_file: Файл с данными для предсказания
            output_file: Файл для сохранения результатов
            model_name: Имя модели
            
        Returns:
            str: Путь к файлу с результатами
        """
        try:
            # Загружаем входные данные
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            results = []
            
            # Обрабатываем каждый элемент
            for item in input_data:
                if 'program_data' in item and 'configurations' in item:
                    program_results = self.predict_multiple_configurations(
                        program_data=item['program_data'],
                        configurations=item['configurations'],
                        model_name=model_name,
                        dvmh_processors=item.get('dvmh_processors', 1)
                    )
                    
                    results.extend(program_results)
                else:
                    self.logger.warning(f"Некорректный формат элемента: {item}")
            
            # Сохраняем результаты
            if output_file is None:
                output_file = f"batch_predictions_{len(results)}_results.json"
            
            output_path = self.save_predictions(results, output_file)
            
            self.logger.info(f"Пакетное предсказание завершено. Результатов: {len(results)}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка при пакетном предсказании: {str(e)}")
            raise
    
    def get_model_info(self, model_name: str = None) -> Dict:
        """
        Получение информации о загруженной модели
        
        Args:
            model_name: Имя модели
            
        Returns:
            Dict: Информация о модели
        """
        if model_name is None:
            if not self.models:
                return {'error': 'Не загружено ни одной модели'}
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            return {'error': f'Модель {model_name} не загружена'}
        
        model_info = self.models[model_name]
        metadata = model_info['metadata']
        
        return {
            'model_name': model_name,
            'model_type': model_info['type'],
            'input_dim': metadata['input_dim'],
            'hidden_dims': metadata['hidden_dims'],
            'metrics': metadata.get('metrics', {}),
            'feature_names': metadata.get('feature_names', []),
            'test_programs': metadata.get('test_programs', []),
            'feature_count': len(metadata.get('feature_names', []))
        }