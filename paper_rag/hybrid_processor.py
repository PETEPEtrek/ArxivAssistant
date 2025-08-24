"""
Гибридный процессор для обработки статей: LaTeX (приоритет) + PDF (резерв)
"""

import logging
from typing import Dict, List, Optional

from .pdf_processor import PDFProcessor
from .latex_processor import LatexProcessor
from ui.arxiv_api import ArxivAPI

logger = logging.getLogger(__name__)

class HybridProcessor:
    """
    Гибридный процессор, который использует LaTeX как приоритетный источник,
    а PDF как резервный для статей, где LaTeX недоступен
    """
    
    def __init__(self):
        """
        Инициализация гибридного процессора
        """
        self.pdf_processor = PDFProcessor()
        self.latex_processor = LatexProcessor()
        self.arxiv_api = ArxivAPI()
        
        # Настройки приоритетов
        self.prefer_latex = True  # Приоритет LaTeX
        self.fallback_to_pdf = True  # Резерв на PDF
        self.force_latex = False  # Принудительно использовать LaTeX
    
    def process_article(self, arxiv_id: str, pdf_url: str = None) -> Optional[Dict]:
        """
        Обработка статьи с приоритетом LaTeX
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            pdf_url: URL для скачивания PDF (резервный вариант)
            
        Returns:
            Результат обработки с метаданными
        """
        logger.info(f"=== Гибридная обработка статьи {arxiv_id} ===")
        
        # Шаг 1: Проверяем доступность LaTeX
        if self.prefer_latex:
            latex_result = self._try_latex_processing(arxiv_id)
            if latex_result:
                logger.info(f"✅ LaTeX обработка успешна для {arxiv_id}")
                return latex_result
        
        # Шаг 2: Если LaTeX недоступен, используем PDF
        if self.fallback_to_pdf and pdf_url:
            logger.info(f"🔄 LaTeX недоступен, переключаемся на PDF для {arxiv_id}")
            pdf_result = self._try_pdf_processing(arxiv_id, pdf_url)
            if pdf_result:
                logger.info(f"✅ PDF обработка успешна для {arxiv_id}")
                return pdf_result
        
        # Шаг 3: Если ничего не работает
        logger.error(f"❌ Не удалось обработать статью {arxiv_id} ни LaTeX, ни PDF")
        return None
    
    def _try_latex_processing(self, arxiv_id: str) -> Optional[Dict]:
        """
        Попытка обработки через LaTeX
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            
        Returns:
            Результат обработки или None
        """
        try:
            logger.info(f"🔍 Пробуем LaTeX обработку для {arxiv_id}")
            
            # Проверяем доступность исходного кода
            available_formats = self.arxiv_api.get_available_formats(arxiv_id)
            if 'source' not in available_formats:
                logger.info(f"Исходный код недоступен для {arxiv_id}")
                return None
            
            # Скачиваем исходный код
            source_path = self.arxiv_api.download_source(arxiv_id, 'source')
            if not source_path:
                logger.warning(f"Не удалось скачать исходный код для {arxiv_id}")
                return None
            
            # Обрабатываем LaTeX
            latex_result = self.latex_processor.extract_from_source(source_path)
            if not latex_result:
                # Проверяем, не является ли файл PDF
                if source_path.endswith('.tar.gz'):
                    logger.warning(f"Файл {source_path} имеет расширение .tar.gz, но не является gzip архивом")
                    logger.info(f"Возможно, это PDF файл с неправильным расширением")
                else:
                    logger.warning(f"Не удалось обработать LaTeX для {arxiv_id}")
                return None
            
            # Добавляем метаданные
            latex_result['metadata']['arxiv_id'] = arxiv_id
            latex_result['metadata']['processing_method'] = 'latex_hybrid'
            latex_result['metadata']['source_file'] = source_path
            
            # Конвертируем в формат, совместимый с RAG
            return self._convert_latex_to_rag_format(latex_result)
            
        except Exception as e:
            logger.error(f"Ошибка при LaTeX обработке {arxiv_id}: {e}")
            return None
    
    def _try_pdf_processing(self, arxiv_id: str, pdf_url: str) -> Optional[Dict]:
        """
        Резервная обработка через PDF
        
        Args:
            arxiv_id: Идентификатор arXiv статьи
            pdf_url: URL для скачивания PDF
            
        Returns:
            Результат обработки или None
        """
        try:
            logger.info(f"📄 PDF резервная обработка для {arxiv_id}")
            
            # Используем существующий PDF процессор
            pdf_result = self.pdf_processor.process_article(arxiv_id, pdf_url)
            if pdf_result:
                pdf_result['metadata']['processing_method'] = 'pdf_fallback'
                return pdf_result
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при PDF обработке {arxiv_id}: {e}")
            return None
    
    def _convert_latex_to_rag_format(self, latex_result: Dict) -> Dict:
        """
        Конвертирует результат LaTeX обработки в формат, совместимый с RAG
        
        Args:
            latex_result: Результат обработки LaTeX
            
        Returns:
            Данные в формате RAG
        """
        # Создаем чанки на основе секций LaTeX
        sections = latex_result.get('sections', [])
        
        # Если секции найдены, используем их для чанкинга
        if sections:
            chunks = self._create_chunks_from_latex_sections(sections, latex_result['metadata'])
            latex_result['chunks'] = chunks
            latex_result['chunking_method'] = 'latex_sections'
        else:
            # Если секции не найдены, разбиваем весь текст
            chunks = self._create_chunks_from_text(latex_result['text'], latex_result['metadata'])
            latex_result['chunks'] = chunks
            latex_result['chunking_method'] = 'text_split'
        
        return latex_result
    
    def _create_chunks_from_latex_sections(self, sections: List[Dict], metadata: Dict) -> List[Dict]:
        """
        Создает чанки на основе секций LaTeX с разбиением больших секций
        
        Args:
            sections: Список секций из LaTeX
            metadata: Метаданные статьи
            
        Returns:
            Список чанков для RAG
        """
        chunks = []
        max_chunk_size = 2000  # Максимальный размер чанка
        chunk_overlap = 200     # Перекрытие между чанками
        
        for i, section in enumerate(sections):
            if not section['text'].strip():  # Пропускаем пустые секции
                continue
            
            section_text = section['text']
            section_title = section['title']
            
            # Если секция слишком большая, разбиваем на подчанки
            if len(section_text) > max_chunk_size:
                section_chunks = self._split_large_section(
                    section_text, section_title, i, metadata, section
                )
                chunks.extend(section_chunks)
            else:
                # Создаем один чанк для маленькой секции
                chunk = {
                    'text': section_text,
                    'metadata': {
                        'arxiv_id': metadata.get('arxiv_id', 'unknown'),
                        'section': section_title,
                        'section_title': section_title,
                        'section_type': section['type'],
                        'section_level': section['level'],
                        'section_index': i,
                        'chunk_type': 'latex_section',
                        'processing_method': 'latex_hybrid',
                        'char_count': section['char_count'],
                        'word_count': section['word_count'],
                        'start_pos': section.get('start_pos', 0),
                        'end_pos': section.get('end_pos', 0),
                        'chunk_index': 0  # Первый (и единственный) чанк секции
                    }
                }
                chunks.append(chunk)
        
        logger.info(f"Создано {len(chunks)} чанков из {len(sections)} LaTeX секций")
        return chunks
    
    def _create_chunks_from_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Создает чанки из полного текста (резервный метод)
        
        Args:
            text: Полный текст статьи
            metadata: Метаданные статьи
            
        Returns:
            Список чанков для RAG
        """
        # Простое разбиение на чанки по размеру
        chunk_size = 1000
        overlap = 200
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Ищем хорошую точку для разрыва
            if end < len(text):
                # Ищем конец предложения или абзаца
                for break_char in ['. ', '\n\n', '! ', '? ']:
                    pos = text.rfind(break_char, start, end)
                    if pos > start + chunk_size // 2:  # Не слишком рано
                        end = pos + len(break_char)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = {
                    'text': chunk_text,
                    'metadata': {
                        'arxiv_id': metadata.get('arxiv_id', 'unknown'),
                        'chunk_type': 'text_split',
                        'processing_method': 'latex_hybrid',
                        'chunk_index': len(chunks),
                        'start_pos': start,
                        'end_pos': end,
                        'char_count': len(chunk_text),
                        'word_count': len(chunk_text.split())
                    }
                }
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        logger.info(f"Создано {len(chunks)} чанков из текста размером {len(text)} символов")
        return chunks
    
    def _split_large_section(self, section_text: str, section_title: str, section_index: int, 
                            metadata: Dict, section_info: Dict) -> List[Dict]:
        """
        Разбивает большую секцию на подчанки
        
        Args:
            section_text: Текст секции
            section_title: Название секции
            section_index: Индекс секции
            metadata: Метаданные статьи
            section_info: Информация о секции
            
        Returns:
            Список подчанков
        """
        max_chunk_size = 2000
        chunk_overlap = 200
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(section_text):
            end = min(start + max_chunk_size, len(section_text))
            
            # Ищем хорошую точку для разрыва
            if end < len(section_text):
                # Ищем конец предложения или абзаца
                for break_char in ['. ', '\n\n', '! ', '? ']:
                    pos = section_text.rfind(break_char, start, end)
                    if pos > start + max_chunk_size // 2:  # Не слишком рано
                        end = pos + len(break_char)
                        break
            
            chunk_text = section_text[start:end].strip()
            if chunk_text:
                chunk = {
                    'text': chunk_text,
                    'metadata': {
                        'arxiv_id': metadata.get('arxiv_id', 'unknown'),
                        'section': section_title,
                        'section_title': section_title,
                        'section_type': section_info['type'],
                        'section_level': section_info['level'],
                        'section_index': section_index,
                        'chunk_type': 'latex_section',
                        'processing_method': 'latex_hybrid',
                        'char_count': len(chunk_text),
                        'word_count': len(chunk_text.split()),
                        'start_pos': section_info.get('start_pos', 0) + start,
                        'end_pos': section_info.get('start_pos', 0) + end,
                        'chunk_index': chunk_index  # Индекс чанка внутри секции
                    }
                }
                chunks.append(chunk)
                chunk_index += 1
            
            start = end - chunk_overlap
            if start >= len(section_text):
                break
        
        logger.info(f"Секция '{section_title}' разбита на {len(chunks)} чанков")
        return chunks