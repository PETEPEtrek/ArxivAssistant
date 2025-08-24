"""
Модуль для асинхронной обработки статей в фоновом режиме
"""

import asyncio
import threading
from typing import Dict, Optional, Callable
import logging
from queue import Queue
import time

from .rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)

class AsyncPaperProcessor:
    """
    Класс для асинхронной обработки статей в фоновом режиме
    """
    
    def __init__(self):
        """
        Инициализация асинхронного процессора
        """
        self.rag_pipeline = rag_pipeline
        self.processing_queue = Queue()
        self.processing_status = {}
        self.active_tasks = {}  # Добавляем активные задачи
        self.worker_thread = None
        self.running = False
        
        # Коллбэки для уведомлений
        self.callbacks = {
            'on_start': [],
            'on_progress': [],
            'on_complete': [],
            'on_error': []
        }
        
        self._start_worker()
    
    @property
    def is_running(self) -> bool:
        """Проверка, работает ли процессор"""
        return self.running and self.worker_thread and self.worker_thread.is_alive()
    
    @property 
    def queue(self) -> Queue:
        """Доступ к очереди обработки"""
        return self.processing_queue
    
    def _start_worker(self):
        """
        Запуск рабочего потока для обработки
        """
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Асинхронный процессор статей запущен")
    
    def _worker_loop(self):
        """
        Основной цикл обработки задач
        """
        while self.running:
            try:
                if not self.processing_queue.empty():
                    task = self.processing_queue.get(timeout=1)
                    self._process_task(task)
                else:
                    time.sleep(0.1)  # Небольшая пауза если нет задач
            except Exception as e:
                logger.error(f"Ошибка в рабочем цикле: {e}")
                time.sleep(1)
    
    def _process_task(self, task: Dict):
        """
        Обработка отдельной задачи
        
        Args:
            task: Словарь с параметрами задачи
        """
        arxiv_id = task['arxiv_id']
        pdf_url = task['pdf_url']
        task_id = task['task_id']
        
        try:
            logger.info(f"Начало асинхронной обработки статьи: {arxiv_id}")
            
            # Обновляем статус
            self.processing_status[task_id] = {
                'status': 'processing',
                'arxiv_id': arxiv_id,
                'stage': 'downloading',
                'progress': 0,
                'start_time': time.time()
            }
            
            # Обновляем активную задачу
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'processing'
            
            # Уведомляем о начале
            self._notify_callbacks('on_start', {
                'task_id': task_id,
                'arxiv_id': arxiv_id
            })
            
            # Этап 1: Скачивание и извлечение текста
            self._update_progress(task_id, 'extracting_text', 25)
            
            # Проверяем, не обработана ли уже статья
            if self._is_article_processed(arxiv_id):
                logger.info(f"Статья {arxiv_id} уже обработана, пропускаем")
                self._complete_task(task_id, {
                    'success': True,
                    'message': 'Статья уже была обработана ранее',
                    'arxiv_id': arxiv_id,
                    'cached': True
                })
                return
            
            # Обрабатываем статью
            result = self.rag_pipeline.process_article(arxiv_id, pdf_url)
            
            if result['success']:
                # Этап 2: Успешное завершение
                self._update_progress(task_id, 'completed', 100)
                
                # Уведомляем об успехе
                self._complete_task(task_id, result)
            else:
                # Обработка ошибки
                self._error_task(task_id, result.get('error', 'Неизвестная ошибка'))
                
        except Exception as e:
            logger.error(f"Ошибка при обработке {arxiv_id}: {e}")
            self._error_task(task_id, str(e))
    
    def _is_article_processed(self, arxiv_id: str) -> bool:
        """
        Проверка, обработана ли уже статья
        
        Args:
            arxiv_id: ID статьи arXiv
            
        Returns:
            True если статья уже в индексе
        """
        try:
            # Проверяем есть ли чанки этой статьи в индексе
            stats = self.rag_pipeline.get_index_status()
            
            if not stats.get('rag_ready', False):
                return False
            
            # Ищем чанки этой статьи
            test_results = self.rag_pipeline.query_processor.search_in_article(
                "test", arxiv_id, k=1
            )
            
            return len(test_results) > 0
            
        except Exception as e:
            logger.warning(f"Ошибка проверки статьи {arxiv_id}: {e}")
            return False
    
    def _update_progress(self, task_id: str, stage: str, progress: int):
        """
        Обновление прогресса задачи
        
        Args:
            task_id: ID задачи
            stage: Текущий этап
            progress: Прогресс в процентах
        """
        if task_id in self.processing_status:
            self.processing_status[task_id].update({
                'stage': stage,
                'progress': progress,
                'updated_time': time.time()
            })
            
            # Уведомляем о прогрессе
            self._notify_callbacks('on_progress', {
                'task_id': task_id,
                'stage': stage,
                'progress': progress
            })
    
    def _complete_task(self, task_id: str, result: Dict):
        """
        Завершение задачи с успехом
        
        Args:
            task_id: ID задачи
            result: Результат обработки
        """
        if task_id in self.processing_status:
            self.processing_status[task_id].update({
                'status': 'completed',
                'stage': 'completed',
                'progress': 100,
                'result': result,
                'completed_time': time.time()
            })
            
            # Уведомляем о завершении
            self._notify_callbacks('on_complete', {
                'task_id': task_id,
                'result': result
            })
            
            # Удаляем из активных задач
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            logger.info(f"Задача {task_id} успешно завершена")
    
    def _error_task(self, task_id: str, error: str):
        """
        Завершение задачи с ошибкой
        
        Args:
            task_id: ID задачи
            error: Описание ошибки
        """
        if task_id in self.processing_status:
            self.processing_status[task_id].update({
                'status': 'error',
                'stage': 'error',
                'progress': 0,
                'error': error,
                'error_time': time.time()
            })
            
            # Удаляем из активных задач
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # Уведомляем об ошибке
            self._notify_callbacks('on_error', {
                'task_id': task_id,
                'error': error
            })
            
            logger.error(f"Задача {task_id} завершена с ошибкой: {error}")
    
    def _notify_callbacks(self, event_type: str, data: Dict):
        """
        Уведомление зарегистрированных коллбэков
        
        Args:
            event_type: Тип события
            data: Данные события
        """
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Ошибка в коллбэке {event_type}: {e}")
    
    def queue_article(self, arxiv_id: str, pdf_url: str) -> str:
        """
        Постановка статьи в очередь на обработку
        
        Args:
            arxiv_id: ID статьи arXiv
            pdf_url: URL для скачивания PDF
            
        Returns:
            ID задачи для отслеживания
        """
        task_id = f"{arxiv_id}_{int(time.time())}"
        
        # Проверяем, не обрабатывается ли уже
        for status in self.processing_status.values():
            if (status.get('arxiv_id') == arxiv_id and 
                status.get('status') in ['queued', 'processing']):
                logger.info(f"Статья {arxiv_id} уже в очереди обработки")
                return status.get('task_id', task_id)
        
        task = {
            'task_id': task_id,
            'arxiv_id': arxiv_id,
            'pdf_url': pdf_url,
            'queued_time': time.time()
        }
        
        # Добавляем в очередь
        self.processing_queue.put(task)
        
        # Обновляем статус
        self.processing_status[task_id] = {
            'status': 'queued',
            'arxiv_id': arxiv_id,
            'stage': 'queued',
            'progress': 0,
            'queued_time': time.time()
        }
        
        # Добавляем в активные задачи
        self.active_tasks[task_id] = {
            'arxiv_id': arxiv_id,
            'status': 'queued',
            'pdf_url': pdf_url,
            'queued_time': time.time()
        }
        
        logger.info(f"Статья {arxiv_id} добавлена в очередь обработки (ID: {task_id})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Получение статуса задачи
        
        Args:
            task_id: ID задачи
            
        Returns:
            Статус задачи или None
        """
        return self.processing_status.get(task_id)
    
    def get_article_status(self, arxiv_id: str) -> Optional[Dict]:
        """
        Получение статуса обработки статьи
        
        Args:
            arxiv_id: ID статьи
            
        Returns:
            Статус последней задачи для статьи
        """
        # Ищем последнюю задачу для этой статьи
        latest_task = None
        latest_time = 0
        
        for task_id, status in self.processing_status.items():
            if (status.get('arxiv_id') == arxiv_id and 
                status.get('queued_time', 0) > latest_time):
                latest_task = status
                latest_time = status.get('queued_time', 0)
        
        return latest_task
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Добавление коллбэка для событий
        
        Args:
            event_type: Тип события (on_start, on_progress, on_complete, on_error)
            callback: Функция коллбэка
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """
        Очистка старых завершенных задач
        
        Args:
            max_age_hours: Максимальный возраст задач в часах
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        tasks_to_remove = []
        for task_id, status in self.processing_status.items():
            task_time = status.get('completed_time') or status.get('error_time')
            if (task_time and 
                current_time - task_time > max_age_seconds and
                status.get('status') in ['completed', 'error']):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.processing_status[task_id]
        
        if tasks_to_remove:
            logger.info(f"Очищено {len(tasks_to_remove)} старых задач")
    
    def stop(self):
        """
        Остановка асинхронного процессора
        """
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Асинхронный процессор остановлен")

# Глобальный экземпляр асинхронного процессора
async_processor = AsyncPaperProcessor()
