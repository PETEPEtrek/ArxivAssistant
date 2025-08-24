"""
Модуль для обработки входящих запросов и поиска релевантных чанков
"""

import re
from typing import Dict, List, Optional, Tuple
import logging

from .embeddings import embedding_manager

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Класс для обработки запросов и поиска релевантной информации
    """
    
    def __init__(self):
        """
        Инициализация процессора запросов
        """
        self.embedding_manager = embedding_manager
        
        # Настройки гибридного поиска (соотношение BM25:Dense = 3:7)
        self.BM25_WEIGHT = 0.3  # 30% веса для BM25 (лексический поиск)
        self.SEMANTIC_WEIGHT = 0.7  # 70% веса для semantic (dense embeddings)
        
        # Словари для улучшения запросов
        self.query_expansions = {
            'что': ['содержание', 'суть', 'описание'],
            'как': ['метод', 'способ', 'подход'],
            'почему': ['причина', 'обоснование'],
            'результат': ['вывод', 'заключение', 'итог'],
            'метод': ['методология', 'подход', 'алгоритм'],
            'эксперимент': ['исследование', 'тест', 'анализ']
        }
        
        # Стоп-слова для фильтрации
        self.stop_words = {
            'что', 'как', 'где', 'когда', 'почему', 'какой', 'какая', 'какое',
            'в', 'на', 'с', 'по', 'для', 'из', 'к', 'от', 'при', 'о', 'об',
            'и', 'или', 'но', 'а', 'да', 'нет', 'не', 'ни', 'же', 'ли',
            'это', 'то', 'та', 'те', 'тот', 'эта', 'эти'
        }
    
    def process_query(self, query: str, arxiv_id: Optional[str] = None) -> Dict:
        """
        Обработка запроса и поиск релевантного чанка
        
        Args:
            query: Пользовательский запрос
            arxiv_id: ID статьи для ограничения поиска (опционально)
            
        Returns:
            Словарь с результатами поиска
        """
        logger.info(f"Обработка запроса: '{query}' для статьи: {arxiv_id}")
        
        # Очистка и улучшение запроса
        processed_query = self.enhance_query(query)
        
        # Поиск релевантного чанка
        top_chunk = self._search_relevant_chunk(processed_query, arxiv_id)
        
        if not top_chunk:
            return {
                'success': False,
                'message': 'Не найдено релевантной информации',
                'query': query,
                'processed_query': processed_query
            }
        
        # Получаем все чанки из той же секции
        section_chunks = self._get_section_chunks(top_chunk, arxiv_id)
        
        
        result = {
            'success': True,
            'query': query,
            'processed_query': processed_query,
            'chunk': top_chunk,  # Самый релевантный чанк
            'section_chunks': section_chunks,  # Все чанки секции
            'context': self._extract_context(top_chunk)
        }
        
        logger.info(f"Найден релевантный чанк с оценкой {top_chunk['score']:.3f}, получено {len(section_chunks)} чанков из секции")
        return result
    
    def enhance_query(self, query: str) -> str:
        """
        Улучшение запроса для лучшего поиска
        
        Args:
            query: Исходный запрос
            
        Returns:
            Улучшенный запрос
        """
        # Приводим к нижнему регистру
        enhanced = query.lower().strip()
        
        # Удаляем лишние символы
        enhanced = re.sub(r'[^\w\s]', ' ', enhanced)
        enhanced = re.sub(r'\s+', ' ', enhanced)
        
        # Расширяем запрос синонимами
        words = enhanced.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in self.query_expansions:
                expanded_words.extend(self.query_expansions[word])
        
        # Удаляем стоп-слова (кроме важных для контекста)
        filtered_words = []
        for word in expanded_words:
            if word not in self.stop_words or len(word) > 3:
                filtered_words.append(word)
        
        enhanced = ' '.join(filtered_words)
        
        logger.debug(f"Запрос улучшен: '{query}' -> '{enhanced}'")
        return enhanced
    
    def _search_relevant_chunk(self, query: str, arxiv_id: Optional[str] = None) -> Optional[Dict]:
        """
        Гибридный поиск релевантного чанка (BM25 + эмбеддинги)
        
        Args:
            query: Обработанный запрос
            arxiv_id: ID статьи для фильтрации
            
        Returns:
            Наиболее релевантный чанк или None
        """
        # Этап 1: BM25 поиск для получения кандидатов
        bm25_candidates = self.embedding_manager.bm25_search(query, k=10)
        
        if not bm25_candidates:
            logger.warning("BM25 поиск не дал результатов, используем только эмбеддинги")
            return self._fallback_embedding_search(query, arxiv_id)
        
        # Фильтруем BM25 кандидатов по arxiv_id если нужно
        if arxiv_id:
            filtered_candidates = []
            for candidate in bm25_candidates:
                chunk_arxiv_id = candidate.get('metadata', {}).get('arxiv_id')
                if chunk_arxiv_id == arxiv_id:
                    filtered_candidates.append(candidate)
            
            if not filtered_candidates:
                logger.warning(f"BM25 не нашел чанков для статьи {arxiv_id}, используем эмбеддинги")
                return self._fallback_embedding_search(query, arxiv_id)
            
            bm25_candidates = filtered_candidates
        
        # Этап 2: Семантическое ранжирование BM25 кандидатов
        best_candidate = self._rerank_with_embeddings(query, bm25_candidates)
        
        # Добавляем отладочную информацию о BM25 кандидатах
        best_candidate['debug_bm25_candidates'] = bm25_candidates[:10]  # Топ-10 BM25 кандидатов
        
        logger.info(f"Гибридный поиск: BM25 нашел {len(bm25_candidates)} кандидатов, лучший после ранжирования: score {best_candidate.get('score', 0):.3f}")
        
        return best_candidate
    
    def _fallback_embedding_search(self, query: str, arxiv_id: Optional[str] = None) -> Optional[Dict]:
        """
        Fallback поиск только по эмбеддингам
        """
        k = 10 if arxiv_id else 5
        results = self.embedding_manager.search(query, k=k)
        
        if not results:
            return None
        
        # Фильтруем по arxiv_id если указан
        if arxiv_id:
            filtered_results = []
            for result in results:
                chunk_arxiv_id = result.get('metadata', {}).get('arxiv_id')
                if chunk_arxiv_id == arxiv_id:
                    filtered_results.append(result)
            
            if filtered_results:
                results = filtered_results
            else:
                logger.warning(f"Не найдено чанков для статьи {arxiv_id}")
                best_result = results[0]  # Возвращаем лучший результат в целом
                best_result['debug_bm25_candidates'] = []
                return best_result
        
        best_result = results[0]
        best_result['debug_bm25_candidates'] = []
        return best_result
    
    def _rerank_with_embeddings(self, query: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Переранжирование BM25 кандидатов с помощью эмбеддингов
        
        Args:
            query: Поисковый запрос
            candidates: Кандидаты от BM25
            
        Returns:
            Лучший кандидат после переранжирования
        """
        if not candidates:
            return None
        
        try:
            # Получаем эмбеддинг запроса
            query_embedding = self.embedding_manager.model.encode([query], convert_to_numpy=True)
            
            # Нормализуем
            import faiss
            faiss.normalize_L2(query_embedding)
            
            best_candidate = None
            best_score = -1
            
            # Переранжируем каждого кандидата
            for candidate in candidates:
                # Получаем эмбеддинг кандидата
                candidate_text = candidate['text']
                candidate_embedding = self.embedding_manager.model.encode([candidate_text], convert_to_numpy=True)
                faiss.normalize_L2(candidate_embedding)
                
                # Вычисляем косинусное сходство
                similarity = float(query_embedding @ candidate_embedding.T)
                
                # Комбинируем BM25 score и semantic score (соотношение 3:7)
                bm25_score = candidate.get('score', 0)
                combined_score = self.BM25_WEIGHT * bm25_score + self.SEMANTIC_WEIGHT * similarity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate.copy()
                    best_candidate['score'] = combined_score
                    best_candidate['bm25_score'] = bm25_score
                    best_candidate['semantic_score'] = similarity
                    best_candidate['search_type'] = 'hybrid'
            
            return best_candidate
            
        except Exception as e:
            logger.error(f"Ошибка переранжирования: {e}")
            # Возвращаем лучший BM25 кандидат
            return candidates[0] if candidates else None

    
    def _extract_context(self, chunk: Dict) -> Dict:
        """
        Извлечение контекстной информации о чанке
        
        Args:
            chunk: Найденный чанк
            
        Returns:
            Контекстная информация
        """
        metadata = chunk.get('metadata', {})
        
        context = {
            'section': metadata.get('section', 'Неизвестно'),
            'arxiv_id': metadata.get('arxiv_id'),
            'chunk_index': metadata.get('chunk_index'),
            'source_method': metadata.get('extraction_method', 'unknown')
        }
        
        # Краткая выжимка из текста (первые 200 символов)
        text = chunk['text']
        if len(text) > 200:
            context['preview'] = text[:200] + "..."
        else:
            context['preview'] = text
        
        return context
    
    def search_in_article(self, query: str, arxiv_id: str, k: int = 3) -> List[Dict]:
        """
        Поиск в конкретной статье с возвращением нескольких результатов
        
        Args:
            query: Поисковый запрос
            arxiv_id: ID статьи arXiv
            k: Количество результатов
            
        Returns:
            Список релевантных чанков
        """
        logger.info(f"Поиск '{query}' в статье {arxiv_id}")
        
        # Получаем больше результатов для фильтрации
        all_results = self.embedding_manager.search(query, k=k*3)
        
        # Фильтруем по статье
        article_results = []
        for result in all_results:
            chunk_arxiv_id = result.get('metadata', {}).get('arxiv_id')
            if chunk_arxiv_id == arxiv_id:
                article_results.append(result)
                if len(article_results) >= k:
                    break
        
        logger.info(f"Найдено {len(article_results)} результатов в статье {arxiv_id}")
        return article_results
    
    def get_article_summary_chunks(self, arxiv_id: str) -> List[Dict]:
        """
        Получение ключевых чанков для краткого изложения статьи
        
        Args:
            arxiv_id: ID статьи arXiv
            
        Returns:
            Список важных чанков
        """
        summary_queries = [
            "abstract introduction summary",
            "conclusion results findings",
            "methodology methods approach"
        ]
        
        summary_chunks = []
        seen_chunks = set()
        
        for query in summary_queries:
            results = self.search_in_article(query, arxiv_id, k=2)
            for result in results:
                chunk_id = result.get('chunk_id')
                if chunk_id not in seen_chunks:
                    summary_chunks.append(result)
                    seen_chunks.add(chunk_id)
        
        # Сортируем по релевантности
        summary_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return summary_chunks[:5]  # Возвращаем топ-5
    
    def _get_section_chunks(self, top_chunk: Dict, arxiv_id: str) -> List[Dict]:
        """
        Получение всех чанков из той же секции, что и найденный релевантный чанк
        
        Args:
            top_chunk: Найденный релевантный чанк
            arxiv_id: ID статьи arXiv
            
        Returns:
            Список всех чанков секции, отсортированных по порядку в документе
        """
        top_chunk_metadata = top_chunk.get('metadata', {})
        section_name = top_chunk_metadata.get('section')
        
        if not section_name or section_name == 'Unknown':
            logger.warning("Не удалось определить секцию для чанка")
            return [top_chunk]
        
        logger.info(f"Ищем все чанки секции '{section_name}' в статье {arxiv_id}")
        
        # Получаем все чанки статьи через поиск с широким запросом
        all_article_chunks = self._get_all_article_chunks(arxiv_id)
        
        if not all_article_chunks:
            logger.warning(f"Не найдено чанков для статьи {arxiv_id}")
            return [top_chunk]
        
        # Фильтруем по секции
        section_chunks = []
        for chunk in all_article_chunks:
            chunk_metadata = chunk.get('metadata', {})
            chunk_section = chunk_metadata.get('section')
            
            if chunk_section == section_name:
                section_chunks.append(chunk)
        
        # Сортируем по порядку в документе (chunk_index)
        section_chunks.sort(key=lambda x: x.get('metadata', {}).get('chunk_index', 0))
        
        logger.info(f"Найдено {len(section_chunks)} чанков в секции '{section_name}'")
        return section_chunks
    
    def _get_all_article_chunks(self, arxiv_id: str) -> List[Dict]:
        """
        Получение всех чанков статьи
        
        Args:
            arxiv_id: ID статьи arXiv
            
        Returns:
            Список всех чанков статьи
        """
        # Используем широкий поиск для получения всех чанков статьи
        # Поищем по общим терминам, которые должны быть в любой научной статье
        broad_queries = [
            "abstract introduction method result conclusion",
            "research study analysis data experiment",
            "paper article work approach model",
            "figure table equation reference"
        ]
        
        all_chunks = []
        seen_chunk_ids = set()
        
        for query in broad_queries:
            # Получаем много результатов для широкого покрытия
            results = self.embedding_manager.search(query, k=50)
            
            for result in results:
                chunk_arxiv_id = result.get('metadata', {}).get('arxiv_id')
                chunk_index = result.get('metadata', {}).get('chunk_index')
                
                # Проверяем, что это нужная статья и мы еще не видели этот чанк
                if (chunk_arxiv_id == arxiv_id and 
                    chunk_index is not None and 
                    chunk_index not in seen_chunk_ids):
                    
                    all_chunks.append(result)
                    seen_chunk_ids.add(chunk_index)
        
        # Сортируем по порядку в документе
        all_chunks.sort(key=lambda x: x.get('metadata', {}).get('chunk_index', 0))
        
        logger.debug(f"Получено {len(all_chunks)} уникальных чанков для статьи {arxiv_id}")
        return all_chunks

# Глобальный экземпляр процессора запросов
query_processor = QueryProcessor()
