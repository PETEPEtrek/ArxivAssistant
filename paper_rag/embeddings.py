"""
Модуль для создания эмбеддингов и работы с FAISS индексом
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Класс для управления эмбеддингами и FAISS индексом
    """
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 embeddings_dir: str = "paper_rag/data/embeddings"):
        """
        Инициализация менеджера эмбеддингов
        
        Args:
            model_name: Название модели для эмбеддингов
            embeddings_dir: Директория для сохранения индекса
        """
        self.model_name = model_name
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.embeddings_dir / "faiss_index.bin"
        self.metadata_path = self.embeddings_dir / "chunks_metadata.pkl"
        self.bm25_path = self.embeddings_dir / "bm25_index.pkl"
        
        self.model = None
        self.index = None
        self.bm25_index = None
        self.chunks_metadata = []
        
        self._initialize_model()
        self._load_existing_index()
    
    def _initialize_model(self):
        """
        Инициализация модели для эмбеддингов
        """
        logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Модель успешно загружена")
    
    def _load_existing_index(self):
        """
        Загрузка существующего индекса и метаданных
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.info("Существующий индекс не найден, будет создан новый")
            return
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS индекс загружен: {self.index.ntotal} векторов")
            
            with open(self.metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
                logger.info(f"Метаданные загружены: {len(self.chunks_metadata)} чанков")
            
            if self.bm25_path.exists() and BM25Okapi:
                with open(self.bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                    logger.info(f"BM25 индекс загружен")
            else:
                logger.info("BM25 индекс не найден, будет создан при следующей индексации")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")
            self.index = None
            self.bm25_index = None
            self.chunks_metadata = []
    
    def create_embeddings(self, chunks: List[Dict]) -> Optional[np.ndarray]:
        """
        Создание эмбеддингов для списка чанков
        
        Args:
            chunks: Список чанков с текстом
            
        Returns:
            Массив эмбеддингов или None при ошибке
        """
        if not self.model:
            logger.error("Модель не инициализирована")
            return None
        
        if not chunks:
            logger.warning("Нет чанков для создания эмбеддингов")
            return None
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            
            logger.info(f"Создание эмбеддингов для {len(texts)} чанков...")
            
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Эмбеддинги созданы: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов: {e}")
            return None
    
    def add_to_index(self, chunks: List[Dict]) -> bool:
        """
        Добавление чанков в FAISS индекс
        
        Args:
            chunks: Список чанков для индексации
            
        Returns:
            True если успешно, False при ошибке
        """

        embeddings = self.create_embeddings(chunks)
        if embeddings is None:
            return False
        
        try:
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                logger.info(f"Создан новый FAISS индекс с размерностью {dimension}")
            
            faiss.normalize_L2(embeddings)
            
            start_id = len(self.chunks_metadata)
            self.index.add(embeddings)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'id': start_id + i,
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'chunk_id': chunk.get('chunk_id', i)
                }
                self.chunks_metadata.append(chunk_metadata)

            self._create_bm25_index()
            
            logger.info(f"Добавлено {len(chunks)} чанков в индекс. Всего: {self.index.ntotal}")

            self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления в индекс: {e}")
            return False
    
    def search(self, query: str, k: int = 1) -> List[Dict]:
        """
        Поиск наиболее похожих чанков
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            Список найденных чанков с оценками релевантности
        """
        if not self.model or not self.index:
            logger.error("Модель или индекс не инициализированы")
            return []
        
        if self.index.ntotal == 0:
            logger.warning("Индекс пуст")
            return []
        
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:
                    continue
                
                chunk_metadata = self.chunks_metadata[idx]
                result = {
                    'text': chunk_metadata['text'],
                    'metadata': chunk_metadata['metadata'],
                    'score': float(score),
                    'rank': i + 1,
                    'chunk_id': chunk_metadata['chunk_id']
                }
                results.append(result)
            
            logger.info(f"Найдено {len(results)} результатов для запроса: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []
    
    
    def _save_index(self):
        """
        Сохранение индекса и метаданных
        """
        try:
            if self.index and faiss:
                faiss.write_index(self.index, str(self.index_path))
                logger.info(f"FAISS индекс сохранен: {self.index_path}")
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
                logger.info(f"Метаданные сохранены: {self.metadata_path}")
            
            if self.bm25_index:
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump(self.bm25_index, f)
                    logger.info(f"BM25 индекс сохранен: {self.bm25_path}")
                
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса: {e}")
    
    def _create_bm25_index(self):
        """
        Создание BM25 индекса из текущих чанков
        """
        if not BM25Okapi:
            logger.warning("rank-bm25 не установлен, пропускаем создание BM25 индекса")
            return
        
        if not self.chunks_metadata:
            logger.warning("Нет чанков для создания BM25 индекса")
            return
        
        try:
            corpus = []
            for chunk in self.chunks_metadata:
                text = chunk['text']
                tokens = text.lower().split()
                corpus.append(tokens)
            
            self.bm25_index = BM25Okapi(corpus)
            logger.info(f"BM25 индекс создан для {len(corpus)} документов")
            
        except Exception as e:
            logger.error(f"Ошибка создания BM25 индекса: {e}")
            self.bm25_index = None
    
    def bm25_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Поиск с использованием BM25
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            Список найденных чанков с BM25 оценками
        """
        if not self.bm25_index:
            logger.warning("BM25 индекс не инициализирован")
            return []
        
        if not self.chunks_metadata:
            return []
        
        try:
            # Токенизируем запрос
            query_tokens = query.lower().split()
            
            # Получаем BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Создаем результаты с индексами
            results_with_scores = [(i, score) for i, score in enumerate(scores)]
            
            # Сортируем по убыванию score
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Берем топ-k результатов
            top_results = results_with_scores[:k]
            
            # Формируем финальные результаты
            results = []
            for idx, score in top_results:
                if idx < len(self.chunks_metadata):
                    chunk_data = self.chunks_metadata[idx]
                    result = {
                        'text': chunk_data['text'],
                        'metadata': chunk_data['metadata'],
                        'score': float(score),
                        'rank': len(results) + 1,
                        'chunk_id': chunk_data['chunk_id'],
                        'search_type': 'bm25'
                    }
                    results.append(result)
            
            logger.info(f"BM25 поиск нашел {len(results)} результатов для запроса: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка BM25 поиска: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """
        Получение статистики индекса
        
        Returns:
            Словарь со статистикой
        """
        stats = {
            'total_chunks': len(self.chunks_metadata),
            'index_size': self.index.ntotal if self.index else 0,
            'model_name': self.model_name,
            'index_exists': self.index is not None,
            'model_loaded': self.model is not None
        }
        
        # Статистика по статьям
        arxiv_ids = set()
        sections = set()
        
        for chunk in self.chunks_metadata:
            metadata = chunk.get('metadata', {})
            if 'arxiv_id' in metadata:
                arxiv_ids.add(metadata['arxiv_id'])
            if 'section' in metadata:
                sections.add(metadata['section'])
        
        stats['unique_articles'] = len(arxiv_ids)
        stats['unique_sections'] = len(sections)
        
        return stats

# Глобальный экземпляр менеджера эмбеддингов
embedding_manager = EmbeddingManager()
