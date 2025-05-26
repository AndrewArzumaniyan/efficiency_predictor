import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
import os
import json
import copy
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .model_trainer import DVMHEfficiencyMLP, DVMHAttentionModel, DVMHModelTrainer
from .feature_preprocessor import DVMHFeaturePreprocessor

logger = logging.getLogger(__name__)


class DVMHModelFineTuner:
    """
    Класс для дообучения (fine-tuning) моделей DVMH
    Поддерживает различные стратегии дообучения и адаптации
    """
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Инициализация системы дообучения
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")
        self.config = self._load_config(config_path)
        
        # Настройки из конфига
        model_config = self.config.get('model_training', {})
        self.models_dir = model_config.get('models_directory', './models')
        self.finetune_dir = os.path.join(self.models_dir, 'finetuned')
        
        # Создаем директорию для дообученных моделей
        os.makedirs(self.finetune_dir, exist_ok=True)
        
        # Настройка устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация компонентов
        self.base_model = None
        self.base_preprocessor = None
        self.base_metadata = None
        self.model_type = None
        
        # Создаем экземпляр базового тренера для использования методов
        self.base_trainer = DVMHModelTrainer(config_path)
        
        self.logger.info(f"DVMHModelFineTuner инициализирован, устройство: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Конфигурационный файл не найден: {config_path}")
            return {}
    
    def load_base_model(self, model_name: str, model_type: str = 'mlp') -> bool:
        """
        Загрузка базовой модели для дообучения
        
        Args:
            model_name: Имя базовой модели
            model_type: Тип модели ('mlp' или 'attention')
            
        Returns:
            bool: True если модель загружена успешно
        """
        self.logger.info(f"Загрузка базовой модели для дообучения: {model_name}")
        
        # Пути к файлам
        model_path = os.path.join(self.models_dir, f"{model_name}.pt")
        metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            self.logger.error(f"Файлы базовой модели не найдены: {model_path}, {metadata_path}")
            return False
        
        try:
            # Загружаем метаданные
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.base_metadata = json.load(f)
            
            # Загружаем checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Извлекаем препроцессор
            self.base_preprocessor = checkpoint['preprocessor']
            
            # Получаем параметры модели
            input_dim = self.base_metadata['input_dim']
            hidden_dims = self.base_metadata['hidden_dims']
            self.model_type = model_type.lower()
            
            # Создаем модель
            if self.model_type == 'attention':
                attention_dim = self.base_metadata.get('attention_dim', 64)
                self.base_model = DVMHAttentionModel(input_dim, hidden_dims, attention_dim)
            else:
                self.base_model = DVMHEfficiencyMLP(input_dim, hidden_dims)
            
            # Загружаем веса
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.base_model.to(self.device)
            
            self.logger.info(f"Базовая модель {model_name} загружена успешно")
            self.logger.info(f"Архитектура: {hidden_dims}, тип: {self.model_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке базовой модели: {str(e)}")
            return False
    
    def prepare_finetuning_data(self, new_df: pd.DataFrame, target_column: str = 'target_speedup',
                               validation_split: float = 0.2, 
                               adaptation_strategy: str = 'extend_preprocessor') -> Dict:
        """
        Подготовка данных для дообучения
        
        Args:
            new_df: Новые данные для дообучения
            target_column: Целевая переменная
            validation_split: Доля валидационной выборки
            adaptation_strategy: Стратегия адаптации ('extend_preprocessor', 'refit_preprocessor', 'use_existing')
            
        Returns:
            Dict: Подготовленные данные
        """
        self.logger.info(f"Подготовка данных для дообучения, стратегия: {adaptation_strategy}")
        
        if self.base_preprocessor is None:
            raise ValueError("Базовая модель не загружена")
        
        new_df_copy = new_df.copy()
        
        # Разделяем на признаки и целевую переменную
        y_new = new_df_copy[target_column].copy()
        
        # Заполняем пропущенные значения в целевой переменной
        if y_new.isna().any():
            target_median = y_new.median()
            y_new = y_new.fillna(target_median)
            self.logger.info(f"Заполнены NaN в целевой переменной медианой: {target_median}")
        
        # Выбираем стратегию адаптации препроцессора
        if adaptation_strategy == 'extend_preprocessor':
            # Расширяем существующий препроцессор новыми категориями
            X_new_processed = self._extend_preprocessor(new_df_copy, target_column)
            
        elif adaptation_strategy == 'refit_preprocessor':
            # Переобучаем препроцессор на новых данных
            self.logger.warning("Переобучение препроцессора может нарушить совместимость с базовой моделью")
            new_preprocessor = DVMHFeaturePreprocessor()
            X_new_processed = new_preprocessor.fit_transform(new_df_copy, target_column)
            self.base_preprocessor = new_preprocessor
            
        elif adaptation_strategy == 'use_existing':
            # Используем существующий препроцессор как есть
            X_new_processed = self.base_preprocessor.transform(new_df_copy)
            
        else:
            raise ValueError(f"Неизвестная стратегия адаптации: {adaptation_strategy}")
        
        # Разделяем на обучающую и валидационную выборки
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_new_processed, y_new, test_size=validation_split, random_state=42
            )
        else:
            X_train, X_val = X_new_processed, None
            y_train, y_val = y_new.values, None
        
        # Преобразуем в нужный формат
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32) if isinstance(y_train, np.ndarray) else y_train.values.astype(np.float32)
        
        if X_val is not None:
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.float32) if isinstance(y_val, np.ndarray) else y_val.values.astype(np.float32)
        
        # Финальная проверка на NaN
        for name, data in [('X_train', X_train), ('y_train', y_train)]:
            if data is not None:
                nan_count = np.isnan(data).sum()
                if nan_count > 0:
                    self.logger.warning(f"NaN в {name}: {nan_count}. Заменяем на 0.")
                    data = np.nan_to_num(data, nan=0.0 if 'X_' in name else 1.0)
        
        if X_val is not None and y_val is not None:
            for name, data in [('X_val', X_val), ('y_val', y_val)]:
                nan_count = np.isnan(data).sum()
                if nan_count > 0:
                    self.logger.warning(f"NaN в {name}: {nan_count}. Заменяем на 0.")
                    data = np.nan_to_num(data, nan=0.0 if 'X_' in name else 1.0)
        
        self.logger.info(f"Данные подготовлены: train={X_train.shape}, val={X_val.shape if X_val is not None else None}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'feature_names': self.base_preprocessor.get_feature_names(),
            'preprocessor': self.base_preprocessor,
            'adaptation_strategy': adaptation_strategy
        }
    
    def _extend_preprocessor(self, df: pd.DataFrame, target_column: str) -> np.ndarray:
        """
        Расширение существующего препроцессора новыми категориями
        
        Args:
            df: Новые данные
            target_column: Целевая переменная
            
        Returns:
            np.ndarray: Обработанные признаки
        """
        self.logger.info("Расширение препроцессора новыми категориями...")
        
        df_work = df.copy()
        
        # Удаляем служебные колонки
        columns_to_drop = [col for col in self.base_preprocessor.exclude_columns + [target_column] 
                          if col in df_work.columns]
        if columns_to_drop:
            df_work = df_work.drop(columns=columns_to_drop)
        
        # Обрабатываем категориальные колонки
        new_categories_found = False
        
        for col in self.base_preprocessor.categorical_columns:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna("unknown")
                
                # Проверяем новые категории
                existing_categories = set(self.base_preprocessor.categorical_mappings[col].keys())
                new_values = set(df_work[col].unique()) - existing_categories
                
                if new_values:
                    new_categories_found = True
                    self.logger.info(f"Найдены новые категории в '{col}': {new_values}")
                    
                    # Расширяем маппинг
                    max_existing_idx = max(self.base_preprocessor.categorical_mappings[col].values())
                    for i, new_value in enumerate(sorted(new_values)):
                        new_idx = max_existing_idx + i + 1
                        self.base_preprocessor.categorical_mappings[col][new_value] = new_idx
                        self.logger.debug(f"Добавлена категория '{new_value}' -> {new_idx}")
        
        if new_categories_found:
            self.logger.info("Препроцессор расширен новыми категориями")
        
        # Применяем препроцессор
        return self.base_preprocessor.transform(df)
    
    def fine_tune_model(self, data: Dict, 
                   fine_tune_strategy: str = 'full_model',
                   learning_rate: float = 0.0001,
                   epochs: int = 50,
                   batch_size: int = 64,
                   patience: int = 10,
                   freeze_layers: Optional[List[str]] = None,
                   weight_decay: float = 0.01) -> Tuple[nn.Module, Dict]:
      """
      Дообучение модели (ИСПРАВЛЕННАЯ ВЕРСИЯ)
      
      Args:
          data: Подготовленные данные
          fine_tune_strategy: Стратегия дообучения ('full_model', 'last_layers', 'adapter', 'gradual_unfreezing')
          learning_rate: Скорость обучения (обычно меньше чем при полном обучении)
          epochs: Количество эпох
          batch_size: Размер батча
          patience: Терпение для ранней остановки
          freeze_layers: Список слоев для заморозки (если нужно)
          weight_decay: Регуляризация
          
      Returns:
          Tuple: Дообученная модель и история обучения
      """
      self.logger.info(f"Начало дообучения, стратегия: {fine_tune_strategy}")
      
      if self.base_model is None:
          raise ValueError("Базовая модель не загружена")
      
      # Создаем копию модели для дообучения
      finetuned_model = copy.deepcopy(self.base_model)
      finetuned_model.train()
      
      # Применяем стратегию дообучения
      self._apply_fine_tune_strategy(finetuned_model, fine_tune_strategy, freeze_layers)
      
      # ИСПРАВЛЕНИЕ: Проверяем что есть обучаемые параметры
      trainable_params = list(filter(lambda p: p.requires_grad, finetuned_model.parameters()))
      if not trainable_params:
          if fine_tune_strategy == 'gradual_unfreezing':
              # Для gradual_unfreezing размораживаем последний слой для начала
              self.logger.warning("Все параметры заморожены, размораживаем последний слой")
              if hasattr(finetuned_model, 'model') and hasattr(finetuned_model.model, '__getitem__'):
                  # Размораживаем последний слой
                  for param in finetuned_model.model[-1].parameters():
                      param.requires_grad = True
                  trainable_params = list(filter(lambda p: p.requires_grad, finetuned_model.parameters()))
          
          if not trainable_params:
              raise ValueError(f"Нет обучаемых параметров для стратегии {fine_tune_strategy}")
      
      self.logger.info(f"Количество обучаемых параметров: {len(trainable_params)}")
      
      # Подготовка данных
      X_train = data['X_train']
      y_train = data['y_train']
      X_val = data['X_val']
      y_val = data['y_val']
      
      # Создаем DataLoader'ы
      train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      
      val_loader = None
      if X_val is not None and y_val is not None:
          val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
          val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
      
      # Настройка оптимизатора с регуляризацией
      optimizer = optim.Adam(
          trainable_params,  # ИСПРАВЛЕНИЕ: Используем только обучаемые параметры
          lr=learning_rate,
          weight_decay=weight_decay
      )
      
      # ИСПРАВЛЕНИЕ: Настройка scheduler без verbose для совместимости
      try:
          # Пробуем с verbose (новые версии PyTorch)
          scheduler = optim.lr_scheduler.ReduceLROnPlateau(
              optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
          )
      except TypeError:
          # Fallback без verbose (старые версии PyTorch)
          self.logger.info("Используется scheduler без verbose (старая версия PyTorch)")
          scheduler = optim.lr_scheduler.ReduceLROnPlateau(
              optimizer, mode='min', factor=0.5, patience=patience//2
          )
      
      criterion = nn.MSELoss()
      
      # История обучения
      history = {
          'train_loss': [],
          'val_loss': [],
          'train_r2': [],
          'val_r2': [],
          'learning_rates': []
      }
      
      best_val_loss = float('inf')
      patience_counter = 0
      best_model_state = None
      
      self.logger.info(f"Начало дообучения на {epochs} эпох...")
      
      for epoch in range(epochs):
          # Обучение
          finetuned_model.train()
          train_losses = []
          train_predictions = []
          train_targets = []
          
          for batch_X, batch_y in train_loader:
              batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
              
              optimizer.zero_grad()
              
              if self.model_type == 'attention':
                  predictions, _ = finetuned_model(batch_X)
              else:
                  predictions = finetuned_model(batch_X)
              
              loss = criterion(predictions, batch_y)
              loss.backward()
              
              # Gradient clipping для стабильности
              torch.nn.utils.clip_grad_norm_(finetuned_model.parameters(), max_norm=1.0)
              
              optimizer.step()
              
              train_losses.append(loss.item())
              train_predictions.extend(predictions.detach().cpu().numpy())
              train_targets.extend(batch_y.detach().cpu().numpy())
          
          # Валидация
          val_loss = 0.0
          val_predictions = []
          val_targets = []
          
          if val_loader is not None:
              finetuned_model.eval()
              with torch.no_grad():
                  for batch_X, batch_y in val_loader:
                      batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                      
                      if self.model_type == 'attention':
                          predictions, _ = finetuned_model(batch_X)
                      else:
                          predictions = finetuned_model(batch_X)
                      
                      loss = criterion(predictions, batch_y)
                      val_loss += loss.item()
                      
                      val_predictions.extend(predictions.cpu().numpy())
                      val_targets.extend(batch_y.cpu().numpy())
              
              val_loss /= len(val_loader)
          else:
              val_loss = np.mean(train_losses)
              val_predictions = train_predictions
              val_targets = train_targets
          
          # Вычисляем метрики
          train_loss = np.mean(train_losses)
          train_r2 = r2_score(train_targets, train_predictions)
          val_r2 = r2_score(val_targets, val_predictions)
          
          # Сохраняем историю
          history['train_loss'].append(train_loss)
          history['val_loss'].append(val_loss)
          history['train_r2'].append(train_r2)
          history['val_r2'].append(val_r2)
          history['learning_rates'].append(optimizer.param_groups[0]['lr'])
          
          # Обновляем learning rate
          scheduler.step(val_loss)
          
          # Логирование
          if epoch % 10 == 0 or epoch < 10:
              self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                            f"train_r2={train_r2:.4f}, val_r2={val_r2:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
          
          # Ранняя остановка
          if val_loss < best_val_loss:
              best_val_loss = val_loss
              patience_counter = 0
              best_model_state = finetuned_model.state_dict().copy()
          else:
              patience_counter += 1
              if patience_counter >= patience:
                  self.logger.info(f"Ранняя остановка на эпохе {epoch}")
                  break
          
          # Градуальное размораживание (если используется)
          if fine_tune_strategy == 'gradual_unfreezing' and epoch > 0 and epoch % 10 == 0:
              self._gradual_unfreeze_fixed(finetuned_model, epoch, optimizer)
      
      # Загружаем лучшие веса
      if best_model_state is not None:
          finetuned_model.load_state_dict(best_model_state)
      
      self.logger.info("Дообучение завершено")
      return finetuned_model, history
    
    def _apply_fine_tune_strategy(self, model: nn.Module, strategy: str, freeze_layers: Optional[List[str]] = None):
        """Применение стратегии дообучения"""
        
        if strategy == 'full_model':
            # Дообучаем всю модель
            for param in model.parameters():
                param.requires_grad = True
            self.logger.info("Стратегия: дообучение всей модели")
            
        elif strategy == 'last_layers':
            # Замораживаем первые слои, дообучаем последние
            if hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
                # Для MLP модели
                total_layers = len(model.model)
                freeze_until = max(1, total_layers - 4)  # Дообучаем последние 4 слоя
                
                for i, layer in enumerate(model.model):
                    if i < freeze_until:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        for param in layer.parameters():
                            param.requires_grad = True
                            
                self.logger.info(f"Стратегия: заморожены первые {freeze_until} слоев")
            else:
                # Для Attention модели
                for name, param in model.named_parameters():
                    if 'embedding' in name or 'query' in name or 'key' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                self.logger.info("Стратегия: заморожены embedding и attention слои")
                
        elif strategy == 'adapter':
            # Замораживаем основную модель, добавляем адаптерные слои
            for param in model.parameters():
                param.requires_grad = False
            
            # Добавляем небольшие обучаемые слои (адаптеры)
            # Это упрощенная версия - в реальности нужно модифицировать архитектуру
            self.logger.info("Стратегия: adapter-based fine-tuning (упрощенная версия)")
            
        elif strategy == 'gradual_unfreezing':
            # Начинаем с замороженной модели
            for param in model.parameters():
                param.requires_grad = False
            self.logger.info("Стратегия: градуальное размораживание (начинаем с замороженной модели)")
            
        # Применяем кастомные настройки заморозки
        if freeze_layers:
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
                    self.logger.debug(f"Заморожен слой: {name}")
    
    def _gradual_unfreeze(self, model: nn.Module, epoch: int):
        """Градуальное размораживание слоев"""
        if hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
            total_layers = len(model.model)
            unfreeze_layer = min(epoch // 10, total_layers - 1)
            
            if unfreeze_layer < total_layers:
                for param in model.model[unfreeze_layer].parameters():
                    param.requires_grad = True
                self.logger.info(f"Разморожен слой {unfreeze_layer}")
    
    def evaluate_finetuned_model(self, model: nn.Module, test_data: Dict) -> Dict:
        """Оценка дообученной модели"""
        self.logger.info("Оценка дообученной модели...")
        
        if 'X_test' not in test_data or 'y_test' not in test_data:
            self.logger.warning("Тестовые данные не предоставлены")
            return {}
        
        model.eval()
        model = model.to(self.device)
        
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            if self.model_type == 'attention':
                predictions, attention_weights = model(X_test_tensor)
                attention_weights = attention_weights.cpu().numpy()
            else:
                predictions = model(X_test_tensor)
                attention_weights = None
            
            predictions = predictions.cpu().numpy()
        
        # Вычисляем метрики
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        self.logger.info("Метрики дообученной модели:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        if self.model_type == 'attention':
            return metrics, attention_weights
        else:
            return metrics
    
    def compare_models(self, original_metrics: Dict, finetuned_metrics: Dict) -> Dict:
        """Сравнение оригинальной и дообученной модели"""
        comparison = {}
        
        for metric in original_metrics:
            if metric in finetuned_metrics:
                original_val = original_metrics[metric]
                finetuned_val = finetuned_metrics[metric]
                
                if metric in ['mse', 'mae', 'rmse']:
                    # Для этих метрик меньше = лучше
                    improvement = ((original_val - finetuned_val) / original_val) * 100
                else:  # r2
                    # Для R² больше = лучше
                    improvement = ((finetuned_val - original_val) / original_val) * 100
                
                comparison[metric] = {
                    'original': original_val,
                    'finetuned': finetuned_val,
                    'improvement_percent': improvement
                }
        
        self.logger.info("Сравнение моделей:")
        for metric, values in comparison.items():
            self.logger.info(f"  {metric}: {values['original']:.4f} -> {values['finetuned']:.4f} "
                           f"({values['improvement_percent']:+.2f}%)")
        
        return comparison
    
    def save_finetuned_model(self, model: nn.Module, base_model_name: str, 
                           suffix: str = None, metadata: Dict = None) -> str:
        """Сохранение дообученной модели"""
        
        # Генерируем имя для дообученной модели
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if suffix:
            finetuned_name = f"{base_model_name}_ft_{suffix}_{timestamp}"
        else:
            finetuned_name = f"{base_model_name}_finetuned_{timestamp}"
        
        # Пути для сохранения
        model_path = os.path.join(self.finetune_dir, f"{finetuned_name}.pt")
        metadata_path = os.path.join(self.finetune_dir, f"{finetuned_name}_metadata.json")
        
        # Сохраняем модель
        torch.save({
            'model_state_dict': model.state_dict(),
            'preprocessor': self.base_preprocessor,
            'base_model_name': base_model_name,
            'fine_tune_timestamp': timestamp
        }, model_path, pickle_protocol=4)
        
        # Дополняем метаданные
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'base_model_name': base_model_name,
            'fine_tune_timestamp': timestamp,
            'model_type': self.base_metadata.get('model_type', 'Unknown'),
            'input_dim': self.base_metadata.get('input_dim', 0),
            'hidden_dims': self.base_metadata.get('hidden_dims', []),
            'feature_names': self.base_preprocessor.get_feature_names(),
            'is_finetuned': True
        })
        
        metadata.update(self.base_preprocessor.get_metadata())
        
        # Сохраняем метаданные
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    
    def plot_fine_tuning_results(self, history: Dict, comparison: Dict = None) -> plt.Figure:
        """Построение графиков результатов дообучения"""
        
        if comparison:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # График потерь
        ax1.plot(history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Fine-tuning Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График R²
        ax2.plot(history['train_r2'], label='Train R²', color='green')
        ax2.plot(history['val_r2'], label='Validation R²', color='orange')
        ax2.set_title('Fine-tuning R² Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(True)
        
        # График learning rate
        ax3.plot(history['learning_rates'], label='Learning Rate', color='purple')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # График сравнения моделей (если есть)
        if comparison:
            metrics = list(comparison.keys())
            original_values = [comparison[m]['original'] for m in metrics]
            finetuned_values = [comparison[m]['finetuned'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, original_values, width, label='Original Model', alpha=0.8)
            ax4.bar(x + width/2, finetuned_values, width, label='Fine-tuned Model', alpha=0.8)
            
            ax4.set_title('Model Comparison')
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Values')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Добавляем проценты улучшения
            for i, metric in enumerate(metrics):
                improvement = comparison[metric]['improvement_percent']
                ax4.text(i, max(original_values[i], finetuned_values[i]) * 1.05, 
                        f'{improvement:+.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def run_comprehensive_fine_tuning(self, new_df: pd.DataFrame, base_model_name: str,
                                    model_type: str = 'mlp',
                                    target_column: str = 'target_speedup',
                                    test_df: pd.DataFrame = None,
                                    fine_tune_strategies: List[str] = None,
                                    validation_split: float = 0.2,
                                    adaptation_strategy: str = 'extend_preprocessor',
                                    learning_rate: float = 0.0001,
                                    epochs: int = 50,
                                    batch_size: int = 64,
                                    patience: int = 10) -> Dict:
        """
        Комплексное дообучение с тестированием разных стратегий
        
        Args:
            new_df: Новые данные для дообучения
            base_model_name: Имя базовой модели
            model_type: Тип модели
            target_column: Целевая переменная
            test_df: Тестовые данные (опционально)
            fine_tune_strategies: Список стратегий для тестирования
            validation_split: Доля валидационной выборки
            adaptation_strategy: Стратегия адаптации препроцессора
            learning_rate: Скорость обучения
            epochs: Количество эпох
            batch_size: Размер батча
            patience: Терпение для ранней остановки
            
        Returns:
            Dict: Результаты дообучения всех стратегий
        """
        self.logger.info("=" * 80)
        self.logger.info("ЗАПУСК КОМПЛЕКСНОГО ДООБУЧЕНИЯ")
        self.logger.info("=" * 80)
        
        # Загрузка базовой модели
        if not self.load_base_model(base_model_name, model_type):
            raise ValueError(f"Не удалось загрузить базовую модель {base_model_name}")
        
        # Подготовка данных
        self.logger.info("Подготовка данных для дообучения...")
        data = self.prepare_finetuning_data(
            new_df, target_column, validation_split, adaptation_strategy
        )
        
        # Подготовка тестовых данных если есть
        test_data = {}
        if test_df is not None:
            self.logger.info("Подготовка тестовых данных...")
            y_test = test_df[target_column].copy()
            if y_test.isna().any():
                y_test = y_test.fillna(y_test.median())
            
            X_test = self.base_preprocessor.transform(test_df)
            test_data = {
                'X_test': X_test.astype(np.float32),
                'y_test': y_test.values.astype(np.float32)
            }
            self.logger.info(f"Тестовые данные подготовлены: {X_test.shape}")
        
        # Определяем стратегии для тестирования
        if fine_tune_strategies is None:
            fine_tune_strategies = ['full_model', 'last_layers', 'gradual_unfreezing']
        
        self.logger.info(f"Тестируемые стратегии: {fine_tune_strategies}")
        
        # Получаем метрики базовой модели на тестовых данных
        base_metrics = {}
        if test_data:
            self.logger.info("Оценка базовой модели...")
            if self.model_type == 'attention':
                base_metrics, _ = self.evaluate_finetuned_model(self.base_model, test_data)
            else:
                base_metrics = self.evaluate_finetuned_model(self.base_model, test_data)
        
        # Результаты всех экспериментов
        results = {
            'base_model_metrics': base_metrics,
            'strategies_results': {},
            'best_strategy': None,
            'best_metrics': None,
            'data_info': {
                'train_size': len(data['X_train']),
                'val_size': len(data['X_val']) if data['X_val'] is not None else 0,
                'test_size': len(test_data['X_test']) if test_data else 0,
                'adaptation_strategy': adaptation_strategy
            }
        }
        
        # Создаем директорию для результатов
        results_dir = os.path.join(self.finetune_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        best_r2 = -float('inf')
        
        # Тестируем каждую стратегию
        for strategy in fine_tune_strategies:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ТЕСТИРОВАНИЕ СТРАТЕГИИ: {strategy.upper()}")
            self.logger.info(f"{'='*60}")
            
            try:
                # Дообучение
                finetuned_model, history = self.fine_tune_model(
                    data=data,
                    fine_tune_strategy=strategy,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience
                )
                
                # Оценка дообученной модели
                finetuned_metrics = {}
                comparison = {}
                
                if test_data:
                    if self.model_type == 'attention':
                        finetuned_metrics, _ = self.evaluate_finetuned_model(finetuned_model, test_data)
                    else:
                        finetuned_metrics = self.evaluate_finetuned_model(finetuned_model, test_data)
                    
                    # Сравнение с базовой моделью
                    comparison = self.compare_models(base_metrics, finetuned_metrics)
                
                # Сохранение модели
                model_name = self.save_finetuned_model(
                    finetuned_model, 
                    base_model_name, 
                    suffix=strategy,
                    metadata={
                        'fine_tune_strategy': strategy,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'metrics': finetuned_metrics,
                        'comparison': comparison
                    }
                )
                
                # Создание графиков
                fig = self.plot_fine_tuning_results(history, comparison)
                fig.savefig(os.path.join(results_dir, f'{strategy}_results.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Сохранение результатов стратегии
                strategy_results = {
                    'model_name': model_name,
                    'history': history,
                    'metrics': finetuned_metrics,
                    'comparison': comparison,
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'final_train_r2': history['train_r2'][-1],
                    'final_val_r2': history['val_r2'][-1]
                }
                
                results['strategies_results'][strategy] = strategy_results
                
                # Определяем лучшую стратегию по R²
                current_r2 = finetuned_metrics.get('r2', -float('inf'))
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    results['best_strategy'] = strategy
                    results['best_metrics'] = finetuned_metrics
                
                self.logger.info(f"Стратегия {strategy} завершена успешно")
                
            except Exception as e:
                self.logger.error(f"Ошибка в стратегии {strategy}: {str(e)}")
                results['strategies_results'][strategy] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Итоговый отчет
        self.logger.info("\n" + "="*80)
        self.logger.info("ИТОГОВЫЙ ОТЧЕТ ДООБУЧЕНИЯ")
        self.logger.info("="*80)
        
        if results['best_strategy']:
            self.logger.info(f"🏆 Лучшая стратегия: {results['best_strategy']}")
            self.logger.info(f"🎯 Лучший R²: {results['best_metrics']['r2']:.4f}")
            
            if base_metrics:
                base_r2 = base_metrics.get('r2', 0)
                improvement = ((results['best_metrics']['r2'] - base_r2) / base_r2) * 100
                self.logger.info(f"📈 Улучшение R²: {improvement:+.2f}%")
        
        self.logger.info("\n📊 Результаты всех стратегий:")
        for strategy, strategy_results in results['strategies_results'].items():
            if 'error' not in strategy_results:
                final_r2 = strategy_results['final_val_r2']
                self.logger.info(f"  {strategy}: R² = {final_r2:.4f}")
            else:
                self.logger.info(f"  {strategy}: ОШИБКА - {strategy_results['error']}")
        
        # Сохранение общего отчета
        report_path = os.path.join(results_dir, f'fine_tuning_report_{base_model_name}.json')
        
        # Подготавливаем данные для JSON (убираем numpy arrays)
        json_results = copy.deepcopy(results)
        for strategy in json_results['strategies_results']:
            if 'history' in json_results['strategies_results'][strategy]:
                del json_results['strategies_results'][strategy]['history']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n📁 Результаты сохранены в: {results_dir}")
        self.logger.info("🎉 Комплексное дообучение завершено!")
        
        return results
    
    def list_finetuned_models(self) -> List[Dict]:
        """Получение списка всех дообученных моделей"""
        if not os.path.exists(self.finetune_dir):
            return []
        
        models = []
        for file in os.listdir(self.finetune_dir):
            if file.endswith('.pt'):
                model_name = file[:-3]
                metadata_path = os.path.join(self.finetune_dir, f"{model_name}_metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        models.append({
                            'name': model_name,
                            'base_model': metadata.get('base_model_name', 'Unknown'),
                            'timestamp': metadata.get('fine_tune_timestamp', 'Unknown'),
                            'strategy': metadata.get('fine_tune_strategy', 'Unknown'),
                            'r2_score': metadata.get('metrics', {}).get('r2', 'Unknown'),
                            'path': os.path.join(self.finetune_dir, file)
                        })
                    except:
                        continue
        
        # Сортируем по времени создания
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models
    
    def _gradual_unfreeze_fixed(self, model: nn.Module, epoch: int, optimizer: optim.Optimizer):
      """Градуальное размораживание слоев (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
      if hasattr(model, 'model') and hasattr(model.model, '__getitem__'):
          total_layers = len(model.model)
          layers_to_unfreeze = min((epoch // 10) + 1, total_layers)  # Размораживаем больше слоев постепенно
          
          # Размораживаем слои с конца
          for i in range(max(0, total_layers - layers_to_unfreeze), total_layers):
              layer_unfrozen = False
              for param in model.model[i].parameters():
                  if not param.requires_grad:
                      param.requires_grad = True
                      layer_unfrozen = True
              
              if layer_unfrozen:
                  self.logger.info(f"Разморожен слой {i}")
          
          # Обновляем оптимизатор с новыми параметрами
          trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
          if trainable_params:
              # Создаем новую группу параметров
              optimizer.param_groups[0]['params'] = trainable_params
              self.logger.debug(f"Обновлен оптимизатор: {len(trainable_params)} параметров")