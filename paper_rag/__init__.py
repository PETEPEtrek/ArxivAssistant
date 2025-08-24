"""
Paper RAG пакет для работы с научными статьями

Модуль предоставляет полный пайплайн для:
1. Загрузки и обработки PDF статей из arXiv
2. Разбиения текста на чанки с метаданными  
3. Создания эмбеддингов и индексации в FAISS
4. Поиска релевантной информации по запросам
"""

from .pdf_processor import pdf_processor
from .chunking import text_chunker
from .embeddings import embedding_manager
from .query_processor import query_processor
from .rag_pipeline import RAGPipeline, rag_pipeline
from .async_processor import async_processor

__version__ = "1.0.0"

__all__ = [
    'pdf_processor',
    'text_chunker', 
    'embedding_manager',
    'query_processor',
    'RAGPipeline',
    'rag_pipeline',
    'async_processor'
]
