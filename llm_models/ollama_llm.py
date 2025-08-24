"""
Класс для работы с локальными моделями через Ollama
"""

import time
import requests
from typing import Dict, Optional, List
import logging

from .base_llm import BaseLLM
from .config import get_ollama_host

try:
    import ollama
    OLLAMA_AVAILABLE = False
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

logger = logging.getLogger(__name__)

class OllamaLLM(BaseLLM):
    """
    Класс для работы с локальными моделями через Ollama
    """
    
    def __init__(self, 
                 model_name: str = "qwen3",
                 host: str = None,
                 timeout: int = 300,
                 **kwargs):
        """
        Инициализация Ollama LLM
        
        Args:
            model_name: Название модели в Ollama (qwen3, mistral, codellama, etc.)
            host: Адрес Ollama сервера (если None, определяется автоматически)
            timeout: Таймаут для запросов
            **kwargs: Дополнительные параметры
        """
        super().__init__(model_name, **kwargs)
        
        # Если хост не указан, определяем автоматически
        self.host = host or get_ollama_host()
        self.timeout = timeout
        
        logger.info(f"Ollama LLM инициализирован с хостом: {self.host}")
        
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama библиотека не установлена, используется requests")
        
        # Проверяем доступность
        self.is_available = self.check_availability()
        
        if self.is_available:
            self._ensure_model_pulled()
    
    def check_availability(self) -> bool:
        """
        Проверка доступности Ollama сервера
        
        Returns:
            True если сервер доступен
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama сервер недоступен: {e}")
            return False
    
    def _ensure_model_pulled(self) -> bool:
        """
        Проверка и загрузка модели если необходимо
        
        Returns:
            True если модель доступна
        """
        try:
            if OLLAMA_AVAILABLE:
                models = ollama.list()
                model_names = []
                for model in models.models:
                    name = getattr(model, 'model', None) or getattr(model, 'name', None)
                    if name:
                        model_names.append(name)
            else:
                response = requests.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    model_names = [model['name'] for model in models_data.get('models', [])]
                else:
                    return False
            
            model_available = False
            for available_model in model_names:
                if (available_model == self.model_name or 
                    available_model.startswith(f"{self.model_name}:") or
                    available_model == f"{self.model_name}:latest"):
                    model_available = True
                    if available_model != self.model_name:
                        logger.info(f"Использую точное имя модели: {available_model} вместо {self.model_name}")
                        self.model_name = available_model
                    break
            
            if not model_available:
                logger.info(f"Модель {self.model_name} не найдена, попытка загрузки...")
                return self._pull_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка проверки модели: {e}")
            return False
    
    def _pull_model(self) -> bool:
        """
        Загрузка модели в Ollama
        
        Returns:
            True если модель успешно загружена
        """
        try:
            logger.info(f"Загрузка модели {self.model_name}...")
            
            if OLLAMA_AVAILABLE:
                ollama.pull(self.model_name)
            else:
                response = requests.post(
                    f"{self.host}/api/pull",
                    json={"name": self.model_name},
                    timeout=300
                )
                
                if response.status_code != 200:
                    logger.error(f"Ошибка загрузки модели: {response.text}")
                    return False
            
            logger.info(f"Модель {self.model_name} успешно загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {self.model_name}: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: float = 0.7) -> Dict:
        """
        Генерация ответа от Ollama модели
        
        Args:
            prompt: Входной промпт
            context: Дополнительный контекст
            max_tokens: Максимальное количество токенов
            temperature: Температура для генерации
            
        Returns:
            Словарь с ответом и метаданными
        """
        if not self.is_available:
            return self.handle_error(
                Exception("Ollama сервер недоступен"), 
                "generate_response"
            )
        
        try:
            start_time = time.time()
            
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            
            # Параметры для генерации
            options = {
                'temperature': temperature,
                'num_predict': max_tokens or 2048,
                'top_p': 0.9,
                'top_k': 40
            }
            
            # Выполняем запрос
            if OLLAMA_AVAILABLE:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=full_prompt,
                    options=options
                )
                content = response['response']
                
            else:
                payload = {
                    'model': self.model_name,
                    'prompt': full_prompt,
                    'stream': False,
                    'options': options
                }
                
                response = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                response_data = response.json()
                content = response_data.get('response', '')
            
            end_time = time.time()
            response_time = end_time - start_time
            
            result = {
                'success': True,
                'content': content,
                'model': self.model_name,
                'tokens_used': None,
                'response_time': response_time,
                'error': None,
                'metadata': {
                    'host': self.host,
                    'local_model': True
                }
            }
            
            logger.info(f"Ollama ответ получен: {response_time:.2f}с")
            return result
            
        except Exception as e:
            return self.handle_error(e, "generate_response")
    
    def generate_chat_response(self, 
                              user_question: str, 
                              article_context: str, 
                              article_metadata: Dict,
                              dialogue_context: Optional[str] = None) -> Dict:
        """
        Генерация ответа для чата по статье
        
        Args:
            user_question: Вопрос пользователя
            article_context: Релевантный контекст из статьи
            article_metadata: Метаданные статьи
            dialogue_context: Контекст предыдущего диалога (опционально)
            
        Returns:
            Ответ модели
        """
        prompt = self.format_chat_prompt(user_question, article_context, article_metadata, dialogue_context)
        return self.generate_response(prompt, max_tokens=1000, temperature=0.3)
    
    def generate_summary(self, article_context: str, article_metadata: Dict) -> Dict:
        """
        Генерация краткого изложения статьи
        
        Args:
            article_context: Контекст из статьи
            article_metadata: Метаданные статьи
            
        Returns:
            Краткое изложение
        """
        prompt = self.format_summary_prompt(article_context, article_metadata)
        return self.generate_response(prompt, max_tokens=800, temperature=0.2)
    
    def get_available_models(self) -> List[str]:
        """
        Получение списка доступных моделей
        
        Returns:
            Список названий моделей
        """
        try:
            if OLLAMA_AVAILABLE:
                models = ollama.list()
                model_names = []
                for model in models.models:
                    name = getattr(model, 'model', None) or getattr(model, 'name', None)
                    if name:
                        model_names.append(name)
                return model_names
            else:
                response = requests.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get('models', [])
                    
                    model_names = []
                    for model in models:
                        if isinstance(model, dict) and 'name' in model:
                            model_names.append(model['name'])
                    
                    return model_names
                else:
                    return []
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            return []
    
    def get_model_info(self) -> Dict:
        """
        Получение информации о модели
        
        Returns:
            Информация о модели
        """
        available_models = self.get_available_models()
        
        return {
            'name': self.model_name,
            'type': 'ollama',
            'provider': 'Ollama (Local)',
            'available': self.is_available,
            'model_installed': self.model_name in available_models,
            'host': self.host,
            'supports_streaming': True,
            'local_inference': True,
            'available_models': available_models,
            'capabilities': [
                'text_generation',
                'chat',
                'summarization',
                'analysis',
                'code_generation'
            ]
        }
    
    def switch_model(self, new_model_name: str) -> bool:
        """
        Переключение на другую модель
        
        Args:
            new_model_name: Название новой модели
            
        Returns:
            True если переключение успешно
        """
        old_model = self.model_name
        self.model_name = new_model_name
        
        if self._ensure_model_pulled():
            logger.info(f"Переключено с {old_model} на {new_model_name}")
            return True
        else:
            self.model_name = old_model
            logger.error(f"Не удалось переключиться на {new_model_name}")
            return False