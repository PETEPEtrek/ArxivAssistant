"""
Class for working with OpenAI API models
"""

import os
import time
from typing import Dict, Optional
import logging

from .base_llm import BaseLLM

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    """
    Class for working with OpenAI API models (GPT-3.5, GPT-4, etc.)
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 organization: Optional[str] = None,
                 **kwargs):
        """
        Initialize OpenAI LLM
        
        Args:
            model_name: Model name (gpt-3.5-turbo, gpt-4, etc.)
            api_key: OpenAI API key
            base_url: Base URL for API (for compatible APIs)
            organization: OpenAI organization ID
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables or parameters")
            return
        
        try:
            client_kwargs = {
                'api_key': self.api_key
            }
            
            if base_url:
                client_kwargs['base_url'] = base_url
            if organization:
                client_kwargs['organization'] = organization
                
            self.client = OpenAI(**client_kwargs)
            self.is_available = self.check_availability()
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.client = None
    
    def check_availability(self) -> bool:
        """
        Check OpenAI API availability
        
        Returns:
            True if API is available
        """
        if not self.client:
            return False
        
        try:
            _ = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
            
        except Exception as e:
            logger.error(f"OpenAI API is not available: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: float = 0.7) -> Dict:
        """
        Generate response from OpenAI model
        
        Args:
            prompt: Input prompt
            context: Additional context (added to system prompt)
            max_tokens: Maximum number of tokens
            temperature: Generation temperature
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.client:
            return self.handle_error(
                Exception("OpenAI клиент не инициализирован"), 
                "generate_response"
            )
        
        try:
            start_time = time.time()
            
            # Формируем сообщения
            messages = []
            
            # Системное сообщение с контекстом если есть
            if context:
                system_content = f"You are an expert assistant for analyzing scientific articles. Use the following context for your answer: {context}"
                messages.append({"role": "system", "content": system_content})
            else:
                messages.append({
                    "role": "system", 
                    "content": "You are an expert assistant for analyzing scientific articles. Answer accurately and professionally."
                })
            
            # Пользовательский промпт
            messages.append({"role": "user", "content": prompt})
            
            # Параметры запроса
            request_params = {
                'model': self.model_name,
                'messages': messages,
                'temperature': temperature
            }
            
            if max_tokens:
                request_params['max_tokens'] = max_tokens
            
            # Выполняем запрос
            response = self.client.chat.completions.create(**request_params)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Извлекаем ответ
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            result = {
                'success': True,
                'content': content,
                'model': self.model_name,
                'tokens_used': tokens_used,
                'response_time': response_time,
                'error': None,
                'metadata': {
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
                    'completion_tokens': response.usage.completion_tokens if response.usage else None
                }
            }
            
            logger.info(f"OpenAI ответ получен: {tokens_used} токенов, {response_time:.2f}с")
            return result
            
        except Exception as e:
            return self.handle_error(e, "generate_response")
    
    def generate_chat_response(self, 
                              user_question: str, 
                              article_context: str, 
                              article_metadata: Dict,
                              dialogue_context: Optional[str] = None) -> Dict:
        """
        Generate response for article chat
        
        Args:
            user_question: User question
            article_context: Relevant context from article
            article_metadata: Article metadata
            dialogue_context: Previous dialogue context (optional)
            
        Returns:
            Model response
        """
        prompt = self.format_chat_prompt(user_question, article_context, article_metadata, dialogue_context)
        return self.generate_response(prompt, max_tokens=1000, temperature=0.3)
    
    def generate_summary(self, article_context: str, article_metadata: Dict) -> Dict:
        """
        Generate article summary
        
        Args:
            article_context: Article context
            article_metadata: Article metadata
            
        Returns:
            Summary
        """
        prompt = self.format_summary_prompt(article_context, article_metadata)
        return self.generate_response(prompt, max_tokens=800, temperature=0.2)
    
    def get_model_info(self) -> Dict:
        """
        Получение информации о модели
        
        Returns:
            Информация о модели
        """
        return {
            'name': self.model_name,
            'type': 'openai',
            'provider': 'OpenAI',
            'available': self.is_available,
            'api_key_set': bool(self.api_key),
            'supports_streaming': True,
            'max_tokens': 4096,
            'capabilities': [
                'text_generation',
                'chat',
                'summarization',
                'analysis'
            ]
        }
