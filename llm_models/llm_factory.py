"""
Фабрика для создания LLM моделей
"""

import os
from typing import Dict, Optional, List
import logging

from .base_llm import BaseLLM
from .openai_llm import OpenAILLM
from .ollama_llm import OllamaLLM, OLLAMA_AVAILABLE

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Фабрика для создания экземпляров LLM моделей
    """
    
    # Предопределенные модели
    PREDEFINED_MODELS = {
        # OpenAI модели
        'gpt-3.5-turbo': {
            'type': 'openai',
            'name': 'gpt-3.5-turbo',
            'description': 'GPT-3.5 Turbo - быстрая и эффективная модель от OpenAI',
            'provider': 'openai'
        },
        'gpt-4': {
            'type': 'openai',
            'name': 'gpt-4',
            'description': 'GPT-4 - мощная модель для сложных задач',
            'provider': 'openai'
        },
        
        'qwen3:latest': {
            'type': 'ollama',
            'name': 'qwen3:latest',
            'description': 'Qwen 3 - мощная многоязычная модель от Alibaba (установлена)',
            'provider': 'ollama'
        }
    }
    
    @classmethod
    def create_llm(cls, 
                   model_type: str,
                   model_name: Optional[str] = None,
                   **kwargs) -> Optional[BaseLLM]:
        """
        Создание экземпляра LLM
        
        Args:
            model_type: Тип модели ('openai' или 'ollama')
            model_name: Название модели
            **kwargs: Дополнительные параметры
            
        Returns:
            Экземпляр LLM или None при ошибке
        """
        try:
            if model_type.lower() == 'openai':
                return cls._create_openai_llm(model_name, **kwargs)
            elif model_type.lower() == 'ollama':
                return cls._create_ollama_llm(model_name, **kwargs)
            else:
                logger.error(f"Неизвестный тип модели: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка создания LLM {model_type}/{model_name}: {e}")
            return None
    
    @classmethod
    def _create_openai_llm(cls, model_name: Optional[str] = None, **kwargs) -> Optional[OpenAILLM]:
        """
        Создание OpenAI LLM
        
        Args:
            model_name: Название модели
            **kwargs: Дополнительные параметры
            
        Returns:
            Экземпляр OpenAILLM или None
        """
        
        model_name = model_name or 'gpt-3.5-turbo'
        
        # Проверяем API ключ
        api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY не найден")
            return None
        
        return OpenAILLM(model_name=model_name, api_key=api_key, **kwargs)
    
    @classmethod
    def _create_ollama_llm(cls, model_name: Optional[str] = None, **kwargs) -> Optional[OllamaLLM]:
        """
        Создание Ollama LLM
        
        Args:
            model_name: Название модели
            **kwargs: Дополнительные параметры
            
        Returns:
            Экземпляр OllamaLLM или None
        """
        model_name = model_name or 'qwen3'
        return OllamaLLM(model_name=model_name, **kwargs)
    
    @classmethod
    def create_from_config(cls, config: Dict) -> Optional[BaseLLM]:
        """
        Создание LLM из конфигурации
        
        Args:
            config: Словарь с конфигурацией
            
        Returns:
            Экземпляр LLM или None
        """
        model_type = config.get('type')
        model_name = config.get('name')
        
        if not model_type:
            logger.error("Тип модели не указан в конфигурации")
            return None
        
        # Убираем type и name из kwargs
        kwargs = {k: v for k, v in config.items() if k not in ['type', 'name']}
        
        return cls.create_llm(model_type, model_name, **kwargs)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[Dict]]:
        """
        Получение списка доступных моделей
        
        Returns:
            Словарь с доступными моделями по типам
        """
        available = {
            'openai': [],
            'ollama': []
        }
        
        # Проверяем OpenAI модели
        if os.getenv('OPENAI_API_KEY'):
            for key, model_info in cls.PREDEFINED_MODELS.items():
                if model_info['type'] == 'openai':
                    available['openai'].append({
                        'key': key,
                        'name': model_info['name'],
                        'description': model_info['description']
                    })
        
        # Проверяем Ollama модели
        try:
            ollama_llm = OllamaLLM()
            if ollama_llm.is_available:
                installed_models = ollama_llm.get_available_models()
                
                # Добавляем предопределенные модели
                for key, model_info in cls.PREDEFINED_MODELS.items():
                    if model_info['type'] == 'ollama':
                        model_status = {
                            'key': key,
                            'name': model_info['name'],
                            'description': model_info['description'],
                            'installed': model_info['name'] in installed_models
                        }
                        available['ollama'].append(model_status)
                
                predefined_names = {info['name'] for info in cls.PREDEFINED_MODELS.values() if info['type'] == 'ollama'}
                for installed_model in installed_models:
                    if installed_model not in predefined_names:
                        model_status = {
                            'key': installed_model,
                            'name': installed_model,
                            'description': f'Установленная модель {installed_model}',
                            'installed': True
                        }
                        available['ollama'].append(model_status)
                        
        except Exception as e:
            logger.warning(f"Ошибка проверки Ollama моделей: {e}")
        
        return available
    
    @classmethod
    def get_recommended_model(cls) -> Optional[str]:
        """
        Получение рекомендуемой модели
        
        Returns:
            Ключ рекомендуемой модели или None
        """
        if os.getenv('OPENAI_API_KEY'):
            return 'gpt-3.5-turbo'
        
        try:
            ollama_llm = OllamaLLM()
            if ollama_llm.is_available:
                available_models = ollama_llm.get_available_models()
                
                if available_models:
                    preferred_order = ['qwen3', 'mistral', 'llama2', 'neural-chat']
                    
                    for model in preferred_order:
                        if model in available_models:
                            return model
                    
                    return available_models[0]
        except Exception as e:
            logger.warning(f"Ошибка проверки рекомендуемой модели: {e}")
        
        return None
    
    @classmethod
    def create_best_available(cls, **kwargs) -> Optional[BaseLLM]:
        """
        Создание лучшей доступной модели
        
        Args:
            **kwargs: Дополнительные параметры
            
        Returns:
            Экземпляр лучшей доступной LLM
        """
        recommended = cls.get_recommended_model()
        if not recommended:
            logger.warning("Нет доступных LLM моделей")
            return None
        
        # Проверяем, есть ли модель в предопределенных
        if recommended in cls.PREDEFINED_MODELS:
            model_config = cls.PREDEFINED_MODELS[recommended]
            return cls.create_llm(
                model_type=model_config['type'],
                model_name=model_config['name'],
                **kwargs
            )
        else:
            # Если модель не предопределена, предполагаем что это Ollama модель
            logger.info(f"Создание непредопределенной Ollama модели: {recommended}")
            return cls.create_llm('ollama', recommended, **kwargs)
    
    @classmethod
    def validate_config(cls, config: Dict) -> Dict:
        """
        Валидация конфигурации модели
        
        Args:
            config: Конфигурация для проверки
            
        Returns:
            Результат валидации с ошибками
        """
        errors = []
        warnings = []
        
        # Проверяем обязательные поля
        if 'type' not in config:
            errors.append("Не указан тип модели")
        elif config['type'] not in ['openai', 'ollama']:
            errors.append(f"Неизвестный тип модели: {config['type']}")
        
        # Проверяем специфичные требования
        if config.get('type') == 'openai':
            if not config.get('api_key') and not os.getenv('OPENAI_API_KEY'):
                errors.append("Не указан API ключ для OpenAI")
        
        elif config.get('type') == 'ollama':
            if not OLLAMA_AVAILABLE:
                warnings.append("Ollama библиотека не установлена, будет использован REST API")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

# Глобальный экземпляр фабрики
llm_factory = LLMFactory()
