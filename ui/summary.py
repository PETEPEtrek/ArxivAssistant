"""
Модуль для суммаризации статей по разделам
"""

import logging
from typing import Dict, List, Optional
from collections import defaultdict
import time

from paper_rag.embeddings import embedding_manager
from .chat import chat_manager

logger = logging.getLogger(__name__)

def summarize_paper_by_sections(arxiv_id: str, progress_bar=None, status_text=None) -> Dict:
    """
    Суммаризация статьи по разделам
    
    Args:
        arxiv_id: ID статьи arXiv
        progress_bar: Streamlit progress bar (опционально)
        status_text: Streamlit text element для статуса (опционально)
        
    Returns:
        Результат суммаризации
    """
    try:
        logger.info(f"Начинаем суммаризацию статьи {arxiv_id}")
        
        # Проверяем наличие индекса
        if not embedding_manager.chunks_metadata:
            return {
                'success': False,
                'error': 'RAG индекс не найден. Сначала обработайте статью.'
            }
        
        # Обновляем статус
        if status_text:
            status_text.text("🔍 Группировка чанков по разделам...")
        if progress_bar:
            progress_bar.progress(0.1)
        
        # Группируем чанки по разделам для данной статьи
        sections_data = _group_chunks_by_sections(arxiv_id)
        
        if not sections_data:
            return {
                'success': False,
                'error': f'Не найдены чанки для статьи {arxiv_id}'
            }
        
        logger.info(f"Найдено {len(sections_data)} разделов для суммаризации")
        
        # Отладочная информация о секциях
        for section_title, chunks in sections_data.items():
            logger.info(f"Секция '{section_title}': {len(chunks)} чанков")
            if chunks:
                first_chunk_metadata = chunks[0].get('metadata', {})
                logger.info(f"  Метаданные первого чанка: {list(first_chunk_metadata.keys())}")
                logger.info(f"  section_title: {first_chunk_metadata.get('section_title', 'N/A')}")
                logger.info(f"  section: {first_chunk_metadata.get('section', 'N/A')}")
        
        # Суммаризируем каждый раздел
        summarized_sections = []
        total_sections = len(sections_data)
        
        for i, (section_title, chunks) in enumerate(sections_data.items()):
            if status_text:
                status_text.text(f"🤖 Суммаризация раздела: {section_title} ({i+1}/{total_sections})")
            
            progress = 0.1 + (i / total_sections) * 0.8
            if progress_bar:
                progress_bar.progress(progress)
            
            # Суммаризируем раздел
            section_summary = _summarize_section(section_title, chunks)
            
            if section_summary:
                summarized_sections.append(section_summary)
            
            # Небольшая пауза чтобы не перегружать LLM
            time.sleep(0.5)
        
        # Завершение
        if status_text:
            status_text.text(f"✅ Суммаризация завершена! Обработано {len(summarized_sections)} разделов")
        if progress_bar:
            progress_bar.progress(1.0)
        
        logger.info(f"Суммаризация завершена: {len(summarized_sections)} разделов")
        
        return {
            'success': True,
            'sections': summarized_sections,
            'total_sections': len(summarized_sections)
        }
        
    except Exception as e:
        logger.error(f"Ошибка суммаризации статьи {arxiv_id}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def _group_chunks_by_sections(arxiv_id: str) -> Dict[str, List[Dict]]:
    """
    Группировка чанков по разделам для конкретной статьи
    
    Args:
        arxiv_id: ID статьи
        
    Returns:
        Словарь: {section_title: [chunks]}
    """
    sections = defaultdict(list)
    
    for chunk in embedding_manager.chunks_metadata:
        chunk_metadata = chunk.get('metadata', {})
        chunk_arxiv_id = chunk_metadata.get('arxiv_id')
        
        # Фильтруем только чанки нужной статьи
        if chunk_arxiv_id == arxiv_id:
            # Поддерживаем как старый формат 'section', так и новый 'section_title' из LaTeX
            section = chunk_metadata.get('section_title') or chunk_metadata.get('section', 'Unknown')
            chunk_index = chunk_metadata.get('chunk_index', 0)
            
            sections[section].append({
                'text': chunk.get('text', ''),
                'metadata': chunk_metadata,
                'chunk_index': chunk_index
            })
    
    # Сортируем чанки в каждом разделе по индексу
    for section_title in sections:
        sections[section_title].sort(key=lambda x: x['chunk_index'])
    
    # Сортируем разделы по первому chunk_index в разделе
    sorted_sections = {}
    section_order = sorted(sections.items(), key=lambda x: min(chunk['chunk_index'] for chunk in x[1]))
    
    for section_title, chunks in section_order:
        sorted_sections[section_title] = chunks
    
    return sorted_sections

def _summarize_section(section_title: str, chunks: List[Dict]) -> Optional[Dict]:
    """
    Суммаризация отдельного раздела
    
    Args:
        section_title: Название раздела
        chunks: Список чанков раздела
        
    Returns:
        Суммаризация раздела или None при ошибке
    """
    try:
        # Объединяем текст всех чанков раздела
        section_texts = []
        for chunk in chunks:
            text = chunk['text'].strip()
            if text:
                section_texts.append(text)
        
        if not section_texts:
            logger.warning(f"Пустой раздел: {section_title}")
            return None
        
        original_text = " ".join(section_texts)
        
        # Ограничиваем длину для LLM (макс 8000 символов)
        max_length = 8000
        if len(original_text) > max_length:
            original_text = original_text[:max_length] + "... [ТЕКСТ ОБРЕЗАН]"
        
        # Создаем метаданные для LLM
        article_metadata = {
            'title': 'Scientific Paper',
            'section': section_title,
            'total_chunks': len(chunks),
            'total_length': len(original_text)
        }
        
        logger.info(f"Суммаризация раздела '{section_title}': {len(chunks)} чанков, {len(original_text)} символов")
        
        # Получаем LLM модель из chat_manager
        llm_model = chat_manager.llm_model
        
        if not llm_model or not llm_model.is_available:
            logger.warning(f"LLM недоступна для суммаризации раздела {section_title}")
            return {
                'title': section_title,
                'summary': f"⚠️ **LLM недоступна.** Раздел содержит {len(chunks)} частей с общим объемом ~{len(original_text)} символов.",
                'original_text': original_text,
                'chunks': chunks,
                'total_length': len(original_text)
            }
        
        # Генерируем суммаризацию через LLM
        llm_response = llm_model.generate_summary(original_text, article_metadata)
        
        if llm_response.get('success'):
            summary = llm_response['content']
        else:
            logger.error(f"Ошибка LLM при суммаризации {section_title}: {llm_response.get('error', 'Unknown')}")
            summary = f"❌ **Ошибка генерации суммаризации.** Раздел содержит {len(chunks)} частей с общим объемом ~{len(original_text)} символов."
        
        return {
            'title': section_title,
            'summary': summary,
            'original_text': original_text,
            'chunks': chunks,
            'total_length': len(original_text)
        }
        
    except Exception as e:
        logger.error(f"Ошибка суммаризации раздела {section_title}: {e}")
        return {
            'title': section_title,
            'summary': f"💥 **Ошибка обработки раздела:** {str(e)}",
            'original_text': " ".join([chunk['text'] for chunk in chunks]),
            'chunks': chunks,
            'total_length': 0
        }
