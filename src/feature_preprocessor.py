import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

logger = logging.getLogger(__name__)


class DVMHFeaturePreprocessor:
    """
    Консистентный препроцессор для признаков DVMH программ
    Обеспечивает идентичную обработку данных при обучении и предсказании
    """
    
    def __init__(self):
        """Инициализация препроцессора"""
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")
        
        # Компоненты препроцессинга
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        # Метаданные для консистентности
        self.feature_order = None  # Порядок признаков
        self.categorical_mappings = {}  # Маппинги для категориальных признаков
        self.categorical_columns = []  # Список категориальных колонок
        self.numeric_columns = []  # Список числовых колонок
        self.exclude_columns = ['parallel_execution_time', 'launch_config', 
                               'normalized_parallel_time', 'efficiency', 
                               'program_name', 'target_speedup']  # Колонки для исключения
        
        # Статистики для заполнения новых категорий
        self.category_fallback_values = {}  # Значения по умолчанию для новых категорий
        
        # Флаг обученности
        self.is_fitted = False
        
        self.logger.info("DVMHFeaturePreprocessor инициализирован")
    
    def fit(self, df: pd.DataFrame, target_column: str = 'target_speedup') -> 'DVMHFeaturePreprocessor':
        """
        Обучение препроцессора на тренировочных данных
        
        Args:
            df: DataFrame с данными для обучения
            target_column: Название целевой переменной
            
        Returns:
            self: Обученный препроцессор
        """
        self.logger.info("Обучение препроцессора...")
        
        df_work = df.copy()
        
        # 1. Удаляем служебные колонки (включая целевую переменную)
        columns_to_drop = [col for col in self.exclude_columns + [target_column] 
                          if col in df_work.columns]
        if columns_to_drop:
            self.logger.debug(f"Удаляем служебные колонки: {columns_to_drop}")
            df_work = df_work.drop(columns=columns_to_drop)
        
        # 2. Определяем типы колонок
        self.categorical_columns = df_work.select_dtypes(include=['object']).columns.tolist()
        if 'program_name' in self.categorical_columns:
            self.categorical_columns.remove('program_name')
        
        # 3. Обрабатываем категориальные колонки
        for col in self.categorical_columns:
            self.logger.debug(f"Обработка категориальной колонки: {col}")
            
            # Заполняем NaN стандартным значением
            df_work[col] = df_work[col].fillna("unknown")
            
            # Создаем детерминистичный маппинг
            unique_values = sorted(df_work[col].dropna().unique())
            self.categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
            
            # Сохраняем значение по умолчанию для новых категорий
            # Используем индекс "unknown" или максимальный индекс + 1
            if "unknown" in self.categorical_mappings[col]:
                self.category_fallback_values[col] = self.categorical_mappings[col]["unknown"]
            else:
                self.category_fallback_values[col] = len(unique_values)
                self.categorical_mappings[col]["__UNKNOWN__"] = self.category_fallback_values[col]
            
            # Применяем маппинг
            df_work[col] = df_work[col].map(self.categorical_mappings[col])
            
            self.logger.debug(f"Создан маппинг для {col}: {len(self.categorical_mappings[col])} категорий")
        
        # 4. Обрабатываем смешанные типы (попытка конвертации в числовой)
        for col in df_work.columns:
            if df_work[col].dtype == 'object' and col not in self.categorical_columns and col != 'program_name':
                self.logger.debug(f"Попытка конвертации смешанного типа: {col}")
                try:
                    # Пытаемся конвертировать в числовой тип
                    df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
                    self.logger.debug(f"Успешно конвертирован в числовой: {col}")
                except:
                    # Если не получается, обрабатываем как категориальный
                    self.logger.debug(f"Обработка как категориальный: {col}")
                    df_work[col] = df_work[col].fillna("unknown")
                    unique_values = sorted(df_work[col].dropna().unique())
                    self.categorical_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
                    self.category_fallback_values[col] = self.categorical_mappings[col].get("unknown", len(unique_values))
                    df_work[col] = df_work[col].map(self.categorical_mappings[col])
                    self.categorical_columns.append(col)
        
        # 5. Определяем числовые колонки
        self.numeric_columns = [col for col in df_work.columns 
                               if col not in self.categorical_columns and col != 'program_name']
        
        # 6. Сохраняем порядок признаков
        self.feature_order = df_work.columns.tolist()
        if 'program_name' in self.feature_order:
            self.feature_order.remove('program_name')
        
        self.logger.info(f"Определен порядок признаков: {len(self.feature_order)} признаков")
        self.logger.debug(f"Категориальные признаки: {len(self.categorical_columns)}")
        self.logger.debug(f"Числовые признаки: {len(self.numeric_columns)}")
        
        # 7. Обучаем imputer и scaler
        features_for_training = df_work[self.feature_order]
        
        # Обучаем imputer
        self.imputer.fit(features_for_training)
        self.logger.debug("Imputer обучен")
        
        # Применяем imputer и обучаем scaler
        features_imputed = self.imputer.transform(features_for_training)
        features_imputed_df = pd.DataFrame(features_imputed, columns=self.feature_order)
        
        self.scaler.fit(features_imputed_df)
        self.logger.debug("Scaler обучен")
        
        # Отмечаем как обученный
        self.is_fitted = True
        
        self.logger.info("Препроцессор обучен успешно")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Применение обученного препроцессора к данным
        
        Args:
            df: DataFrame с данными для преобразования
            
        Returns:
            np.ndarray: Преобразованные признаки
        """
        if not self.is_fitted:
            raise ValueError("Препроцессор не обучен. Сначала вызовите fit()")
        
        self.logger.debug("Применение препроцессора...")
        
        df_work = df.copy()
        
        # 1. Удаляем служебные колонки
        columns_to_drop = [col for col in self.exclude_columns if col in df_work.columns]
        if columns_to_drop:
            df_work = df_work.drop(columns=columns_to_drop)
        
        # 2. Обрабатываем категориальные колонки
        for col in self.categorical_columns:
            if col in df_work.columns:
                # Заполняем NaN
                df_work[col] = df_work[col].fillna("unknown")
                
                # Применяем сохраненный маппинг
                def map_category(value):
                    if value in self.categorical_mappings[col]:
                        return self.categorical_mappings[col][value]
                    else:
                        # Новая категория - используем fallback значение
                        self.logger.debug(f"Новая категория '{value}' в колонке '{col}', используем fallback")
                        return self.category_fallback_values[col]
                
                df_work[col] = df_work[col].apply(map_category)
        
        # 3. Обрабатываем смешанные типы как при обучении
        for col in df_work.columns:
            if df_work[col].dtype == 'object' and col not in self.categorical_columns and col != 'program_name':
                if col in self.categorical_mappings:
                    # Это колонка, которая была обработана как категориальная при обучении
                    df_work[col] = df_work[col].fillna("unknown")
                    df_work[col] = df_work[col].apply(
                        lambda x: self.categorical_mappings[col].get(x, self.category_fallback_values[col])
                    )
                else:
                    # Пытаемся конвертировать в числовой
                    try:
                        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
                    except:
                        # Если не получается, заполняем нулями
                        df_work[col] = 0
        
        # 4. Обеспечиваем правильный порядок признаков
        missing_features = set(self.feature_order) - set(df_work.columns)
        extra_features = set(df_work.columns) - set(self.feature_order) - {'program_name'}
        
        # Добавляем отсутствующие признаки (заполняем нулями)
        for feature in missing_features:
            df_work[feature] = 0
            self.logger.debug(f"Добавлен отсутствующий признак '{feature}' (заполнен нулями)")
        
        # Удаляем лишние признаки
        for feature in extra_features:
            if feature in df_work.columns:
                df_work = df_work.drop(columns=[feature])
                self.logger.debug(f"Удален лишний признак '{feature}'")
        
        # Убираем program_name если есть
        if 'program_name' in df_work.columns:
            df_work = df_work.drop(columns=['program_name'])
        
        # 5. Применяем строгий порядок признаков
        df_work = df_work[self.feature_order]
        
        # 6. Применяем imputer и scaler
        features_imputed = self.imputer.transform(df_work)
        features_scaled = self.scaler.transform(features_imputed)
        
        # 7. Финальная обработка
        features_array = features_scaled.astype(np.float32)
        
        # Заменяем NaN на 0 (на всякий случай)
        nan_count = np.isnan(features_array).sum()
        if nan_count > 0:
            self.logger.warning(f"Найдено NaN значений: {nan_count}. Заменяем на 0.")
            features_array = np.nan_to_num(features_array, nan=0.0)
        
        self.logger.debug(f"Преобразование завершено. Размерность: {features_array.shape}")
        return features_array
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'target_speedup') -> np.ndarray:
        """
        Обучение и применение препроцессора за один вызов
        
        Args:
            df: DataFrame с данными
            target_column: Название целевой переменной
            
        Returns:
            np.ndarray: Преобразованные признаки
        """
        return self.fit(df, target_column).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Получение списка имен признаков в правильном порядке"""
        if not self.is_fitted:
            raise ValueError("Препроцессор не обучен")
        return self.feature_order.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Получение метаданных препроцессора для сохранения
        
        Returns:
            Dict: Метаданные препроцессора
        """
        if not self.is_fitted:
            raise ValueError("Препроцессор не обучен")
        
        return {
            'feature_order': self.feature_order,
            'categorical_mappings': self.categorical_mappings,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'category_fallback_values': self.category_fallback_values,
            'exclude_columns': self.exclude_columns,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_order)
        }
    
    def save(self, filepath: str) -> None:
        """
        Сохранение препроцессора в файл
        
        Args:
            filepath: Путь для сохранения
        """
        if not self.is_fitted:
            raise ValueError("Препроцессор не обучен")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"Препроцессор сохранен в: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DVMHFeaturePreprocessor':
        """
        Загрузка препроцессора из файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            DVMHFeaturePreprocessor: Загруженный препроцессор
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info(f"Препроцессор загружен из: {filepath}")
        return preprocessor
    
    def __getstate__(self):
        """Подготовка объекта для сериализации"""
        state = self.__dict__.copy()
        # Удаляем logger, так как он не сериализуется
        if 'logger' in state:
            del state['logger']
        return state
    
    def __setstate__(self, state):
        """Восстановление объекта после десериализации"""
        self.__dict__.update(state)
        # Восстанавливаем logger
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")