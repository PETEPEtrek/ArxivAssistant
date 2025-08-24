"""
Основной RAG пайплайн для работы с научными статьями
"""
import os
import logging
from typing import Dict, List, Optional
from pathlib import Path

from .pdf_processor import pdf_processor
from .chunking import text_chunker
from .embeddings import embedding_manager
from .query_processor import query_processor
from .hybrid_processor import HybridProcessor

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Основной класс для RAG пайплайна работы с научными статьями
    """
    
    def __init__(self, use_hybrid_processor: bool = True):
        """
        Инициализация RAG пайплайна
        
        Args:
            use_hybrid_processor: Использовать ли гибридный процессор (LaTeX + PDF)
        """
        self.use_hybrid_processor = use_hybrid_processor
        
        if use_hybrid_processor:
            self.hybrid_processor = HybridProcessor()
            logger.info("Инициализирован гибридный процессор (LaTeX + PDF)")
        else:
            self.hybrid_processor = None
            logger.info("Используется стандартный PDF процессор")
        
        self.pdf_processor = pdf_processor
        self.text_chunker = text_chunker
        self.embedding_manager = embedding_manager
        self.query_processor = query_processor
        
        debug_mode = os.getenv('RAG_DEBUG', 'false').lower() == 'true'
        log_level = logging.DEBUG if debug_mode else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process_article(self, arxiv_id: str, pdf_url: str = None) -> Dict:
        """
        Полный пайплайн обработки статьи: от LaTeX/PDF до индексации
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            pdf_url: URL для скачивания PDF (резервный вариант)
            
        Returns:
            Результат обработки с статистикой
        """
        logger.info(f"=== Начало обработки статьи {arxiv_id} ===")
        
        try:
            # Шаг 1: Обработка статьи (LaTeX или PDF)
            if self.use_hybrid_processor and self.hybrid_processor:
                logger.info("Шаг 1: Гибридная обработка (LaTeX приоритет, PDF резерв)")
                extracted_data = self.hybrid_processor.process_article(arxiv_id, pdf_url)
            else:
                logger.info("Шаг 1: Стандартная PDF обработка")
                extracted_data = self.pdf_processor.process_article(arxiv_id, pdf_url)
            
            if not extracted_data:
                return {
                    'success': False,
                    'error': 'Не удалось обработать статью',
                    'arxiv_id': arxiv_id
                }
            
            # Шаг 2: Разбиение на чанки
            logger.info("Шаг 2: Разбиение текста на чанки")
            
            if self.use_hybrid_processor and extracted_data.get('chunks'):
                # Если чанки уже созданы гибридным процессором
                logger.info("Используем предсозданные чанки из гибридного процессора")
                chunks = extracted_data['chunks']
                chunking_method = extracted_data.get('chunking_method', 'hybrid')
            else:
                # Используем стандартный чанкер
                logger.info("Используем стандартный чанкер")
                chunks = self.text_chunker.chunk_text(extracted_data)
                chunking_method = 'standard'
            
            if not chunks:
                return {
                    'success': False,
                    'error': 'Не удалось создать чанки из текста',
                    'arxiv_id': arxiv_id
                }
            
            # Шаг 3: Создание эмбеддингов и добавление в индекс
            logger.info("Шаг 3: Создание эмбеддингов и индексация")
            success = self.embedding_manager.add_to_index(chunks)
            
            if not success:
                return {
                    'success': False,
                    'error': 'Не удалось добавить в индекс',
                    'arxiv_id': arxiv_id
                }
            
            # Статистика обработки
            processing_method = extracted_data['metadata'].get('processing_method', 'unknown')
            total_text_length = len(extracted_data.get('text', ''))
            
            result = {
                'success': True,
                'arxiv_id': arxiv_id,
                'text_length': total_text_length,
                'characters_processed': total_text_length,
                'chunks_created': len(chunks),
                'extraction_method': extracted_data['metadata'].get('extraction_method', 'unknown'),
                'processing_method': processing_method,
                'chunking_method': chunking_method,
                'sections_found': len(extracted_data.get('sections', [])),
                'index_stats': self.embedding_manager.get_index_stats()
            }
            
            # Добавляем специфичную информацию для LaTeX
            if processing_method == 'latex_hybrid':
                result['latex_sections'] = len(extracted_data.get('sections', []))
                result['latex_structure'] = {
                    'title': extracted_data.get('structure', {}).get('title'),
                    'authors_count': len(extracted_data.get('structure', {}).get('authors', [])),
                    'environments_count': len(extracted_data.get('structure', {}).get('environments', []))
                }
            
            logger.info(f"=== Статья {arxiv_id} успешно обработана ===")
            logger.info(f"Метод: {processing_method}, Чанки: {len(chunks)}, Чанкинг: {chunking_method}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки статьи {arxiv_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'arxiv_id': arxiv_id
            }
    
    def query_article(self, query: str, arxiv_id: Optional[str] = None) -> Dict:
        """
        Поиск ответа на вопрос в статье или всей базе
        
        Args:
            query: Вопрос пользователя
            arxiv_id: ID конкретной статьи (опционально)
            
        Returns:
            Результат поиска с контекстом
        """
        logger.info(f"Поиск ответа на вопрос: '{query}'")
        
        if arxiv_id:
            logger.info(f"Поиск ограничен статьей: {arxiv_id}")
        
        try:
            # Обрабатываем запрос
            result = self.query_processor.process_query(query, arxiv_id)
            
            if result['success']:
                # Добавляем дополнительную информацию
                chunk = result['chunk']
                section_chunks = result.get('section_chunks', [chunk])
                context = result['context']
                
                # Формируем ответ на основе всей секции
                answer = self._generate_answer_from_section(query, chunk, section_chunks, context)
                result['answer'] = answer
                
                #logger.info(f"Найден ответ с релевантностью: {result['relevance']['level']}, использовано {len(section_chunks)} чанков из секции")
            else:
                logger.warning(f"Не найден релевантный ответ на вопрос: {query}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при поиске ответа: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def _generate_answer_from_section(self, query: str, top_chunk: Dict, section_chunks: List[Dict], context: Dict) -> str:
        """
        Генерация ответа на основе всех чанков секции
        
        Args:
            query: Исходный вопрос
            top_chunk: Самый релевантный чанк
            section_chunks: Все чанки из той же секции
            context: Контекстная информация
            
        Returns:
            Сформированный ответ из всей секции
        """
        section = context.get('section', 'Неизвестный раздел')
        relevance = top_chunk.get('score', 0)
        
        # Формируем ответ
        answer_parts = []
        
        # Заголовок с информацией о секции
        answer_parts.append(f"**Ответ из раздела '{section}' (полный контекст секции):**")
        
        # Объединяем тексты всех чанков секции в правильном порядке
        section_text_parts = []
        for chunk in section_chunks:
            text = chunk['text'].strip()
            if text and text not in section_text_parts:  # Избегаем дублирования
                section_text_parts.append(text)
        
        # Объединяем тексты
        full_section_text = " ".join(section_text_parts)
        
        # Если текст очень длинный, ограничиваем его
        max_length = 4000  # Максимальная длина для LLM
        if len(full_section_text) > max_length:
            top_chunk_text = top_chunk['text']
            remaining_length = max_length - len(top_chunk_text) - 100
            
            if remaining_length > 0:
                truncated_text = full_section_text[:remaining_length] + "..."
                full_section_text = truncated_text + "\n\n[MOST RELEVANT PART]\n" + top_chunk_text
            else:
                full_section_text = top_chunk_text
            
            answer_parts.append(f"*Примечание: Текст секции сокращен. Показано {len(section_chunks)} чанков.*")
        
        answer_parts.append(full_section_text)
        
        answer_parts.append(f"\n*Источник: раздел '{section}'*")
        
        #answer_parts.append(f"*Релевантность: {relevance:.2%}, использовано {len(section_chunks)} чанков*")
        
        return "\n\n".join(answer_parts)
    
    def get_article_summary(self, arxiv_id: str) -> Dict:
        """
        Получение краткого изложения статьи
        
        Args:
            arxiv_id: ID статьи arXiv
            
        Returns:
            Краткое изложение с ключевыми частями
        """
        logger.info(f"Создание краткого изложения для статьи: {arxiv_id}")
        
        try:
            # Получаем ключевые чанки
            summary_chunks = self.query_processor.get_article_summary_chunks(arxiv_id)
            
            if not summary_chunks:
                return {
                    'success': False,
                    'error': 'Не найдены чанки для статьи',
                    'arxiv_id': arxiv_id
                }
            
            # Группируем по разделам
            sections_summary = {}
            for chunk in summary_chunks:
                section = chunk.get('metadata', {}).get('section', 'Unknown')
                if section not in sections_summary:
                    sections_summary[section] = []
                sections_summary[section].append(chunk['text'])
            
            return {
                'success': True,
                'arxiv_id': arxiv_id,
                'sections_summary': sections_summary,
                'total_chunks': len(summary_chunks)
            }
            
        except Exception as e:
            logger.error(f"Ошибка создания изложения: {e}")
            return {
                'success': False,
                'error': str(e),
                'arxiv_id': arxiv_id
            }
    
    def get_index_status(self) -> Dict:
        """
        Получение статуса RAG системы
        
        Returns:
            Статистика и статус всех компонентов
        """
        try:
            stats = self.embedding_manager.get_index_stats()
            
            status = {
                'rag_ready': stats['model_loaded'] and stats['index_exists'],
                'components': {
                    'pdf_processor': True,  # Всегда готов
                    'text_chunker': True,   # Всегда готов
                    'embedding_manager': stats['model_loaded'],
                    'query_processor': stats['model_loaded']
                },
                'index_stats': stats,
                'data_directories': {
                    'papers': str(self.pdf_processor.data_dir),
                    'embeddings': str(self.embedding_manager.embeddings_dir)
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Ошибка получения статуса: {e}")
            return {
                'rag_ready': False,
                'error': str(e)
            }

# Глобальный экземпляр RAG пайплайна
rag_pipeline = RAGPipeline()
