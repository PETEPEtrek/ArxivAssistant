"""
Модуль для чанкинга текста по разделам (заголовкам)
"""

import re
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .visual_chunking import visual_header_detector

logger = logging.getLogger(__name__)

class SectionBasedChunker:
    """
    Класс для разбиения текста на чанки по разделам статьи
    """
    
    def __init__(self, 
                 max_section_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Инициализация чанкера
        
        Args:
            max_section_size: Максимальный размер раздела в символах
            chunk_overlap: Перекрытие для больших разделов
        """
        self.max_section_size = max_section_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_section_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    
    def extract_sections(self, text: str, pdf_path: Optional[str] = None) -> List[Dict]:
        """
        Извлечение разделов из текста научной статьи
        
        Args:
            text: Полный текст статьи
            pdf_path: Путь к PDF файлу для визуального анализа (опционально)
            
        Returns:
            Список разделов с заголовками и содержимым
        """
        logger.info("Извлечение разделов из текста")
        
        # Пробуем визуальный анализ, если доступен PDF файл
        if pdf_path:
            headers = self._extract_visual_headers(text, pdf_path)
            if headers and len(headers) >= 8:  # Если найдено достаточно заголовков
                logger.info(f"Использован визуальный анализ, найдено {len(headers)} заголовков")
            else:
                logger.info(f"Визуальный анализ нашел только {len(headers)} заголовков, используем комбинированный подход")
                # Комбинируем визуальные и regex заголовки
                regex_headers = self._find_all_headers(text)
                headers = self._combine_headers(headers, regex_headers, text)
                logger.info(f"Комбинированный анализ: {len(headers)} заголовков")
        else:
            # Используем старый метод с regex паттернами
            headers = self._find_all_headers(text)
        
        if not headers:
            logger.warning("Заголовки не найдены, создаем один общий раздел")
            return [{
                'title': 'Full Article',
                'content': text,
                'start_pos': 0,
                'end_pos': len(text),
                'level': 0
            }]
        
        # Создаем разделы на основе заголовков
        sections = self._create_sections_from_headers(text, headers)
        
        logger.info(f"Найдено {len(sections)} разделов")
        
        # Возвращаем только секции (чанки будут созданы в chunking.py)
        return sections
    
    def _combine_headers(self, visual_headers: List[Dict], regex_headers: List[Dict], text: str) -> List[Dict]:
        """
        Комбинирование визуальных и regex заголовков
        
        Args:
            visual_headers: Заголовки из визуального анализа
            regex_headers: Заголовки из regex анализа
            text: Полный текст статьи
            
        Returns:
            Объединенный список заголовков
        """
        combined = []
        
        # Добавляем все визуальные заголовки
        for vh in visual_headers:
            combined.append(vh)
        
        # Добавляем regex заголовки, которых нет среди визуальных
        visual_titles = {h['title'].lower() for h in visual_headers}
        
        for rh in regex_headers:
            title_lower = rh['title'].lower()
            
            # Проверяем, нет ли уже похожего заголовка
            is_duplicate = False
            for vt in visual_titles:
                if (title_lower == vt or 
                    title_lower in vt or 
                    vt in title_lower or
                    abs(rh['start_pos'] - next((vh['start_pos'] for vh in visual_headers if vh['title'].lower() == vt), -1000)) < 100):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined.append(rh)
        
        # Сортируем по позиции в тексте
        combined.sort(key=lambda x: x['start_pos'])
        
        # Удаляем дубликаты по позиции (если два заголовка очень близко)
        filtered = []
        for header in combined:
            is_close_duplicate = False
            for existing in filtered:
                if abs(header['start_pos'] - existing['start_pos']) < 50:
                    # Оставляем тот, который лучше (визуальный приоритетнее)
                    if 'font_size' in header and 'font_size' not in existing:
                        # Заменяем regex на visual
                        filtered.remove(existing)
                        break
                    else:
                        is_close_duplicate = True
                        break
            
            if not is_close_duplicate:
                filtered.append(header)
        
        logger.info(f"Комбинирование заголовков: {len(visual_headers)} визуальных + {len(regex_headers)} regex = {len(filtered)} итого")
        return filtered
    
    def chunk_sections(self, sections: List[Dict], arxiv_id: str, page_info: Optional[Dict] = None) -> List[Dict]:
        """
        Создание чанков из разделов
        
        Args:
            sections: Список разделов
            arxiv_id: ID статьи arXiv
            page_info: Информация о страницах (опционально)
            
        Returns:
            Список чанков с метаданными
        """
        logger.info(f"Создание чанков из {len(sections)} разделов")
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_title = section['title']
            section_content = section['content']
            
            # Проверяем размер раздела
            if len(section_content) <= self.max_section_size:
                # Раздел помещается в один чанк
                chunk = self._create_chunk(
                    text=section_content,
                    section_title=section_title,
                    chunk_index=chunk_index,
                    arxiv_id=arxiv_id,
                    section_start_pos=section.get('start_pos', 0),
                    page_info=page_info
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Раздел слишком большой, разбиваем на подчанки
                sub_chunks = self._split_large_section(
                    section_content,
                    section_title,
                    chunk_index,
                    arxiv_id,
                    section.get('start_pos', 0),
                    page_info
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
        
        logger.info(f"Создано {len(chunks)} чанков")
        return chunks
    
    def _extract_visual_headers(self, text: str, pdf_path: str) -> List[Dict]:
        """
        Извлечение заголовков с использованием визуального анализа PDF
        
        Args:
            text: Полный текст статьи
            pdf_path: Путь к PDF файлу
            
        Returns:
            Список заголовков найденных визуальным анализом
        """
        try:
            # Получаем визуальные заголовки из PDF
            visual_headers = visual_header_detector.extract_visual_headers(pdf_path)
            
            if not visual_headers:
                return []
            
            # Сопоставляем визуальные заголовки с позициями в тексте
            matched_headers = []
            
            for vh in visual_headers:
                header_text = vh['text'].strip()
                
                # Ищем этот текст в общем тексте статьи
                text_position = self._find_text_position(text, header_text)
                
                if text_position is not None:
                    matched_headers.append({
                        'title': header_text,
                        'start_pos': text_position,
                        'end_pos': text_position + len(header_text),
                        'match_start': text_position,
                        'match_end': text_position + len(header_text),
                        'level': self._determine_visual_header_level(vh),
                        'font_size': vh.get('font_size', 0),
                        'is_bold': vh.get('is_bold', False),
                        'score': vh.get('score', 0)
                    })
            
            # Сортируем по позиции в тексте
            matched_headers.sort(key=lambda x: x['start_pos'])
            
            logger.info(f"Сопоставлено {len(matched_headers)} визуальных заголовков с текстом")
            return matched_headers
            
        except Exception as e:
            logger.error(f"Ошибка визуального анализа заголовков: {e}")
            return []
    
    def _find_text_position(self, text: str, header_text: str) -> Optional[int]:
        """
        Поиск позиции заголовка в тексте
        
        Args:
            text: Полный текст
            header_text: Текст заголовка
            
        Returns:
            Позиция заголовка в тексте или None
        """
        # Сначала точное совпадение
        pos = text.find(header_text)
        if pos != -1:
            return pos
        
        # Поиск с небольшими вариациями (убираем лишние пробелы, переносы)
        normalized_header = re.sub(r'\s+', ' ', header_text.strip())
        normalized_text = re.sub(r'\s+', ' ', text)
        
        pos = normalized_text.find(normalized_header)
        if pos != -1:
            # Переводим позицию обратно в оригинальный текст (приблизительно)
            return pos
        
        # Поиск по словам (если заголовок состоит из нескольких слов)
        words = header_text.split()
        if len(words) > 1:
            # Ищем первое слово, затем проверяем что следующие слова идут рядом
            first_word = words[0]
            start_pos = 0
            
            while True:
                pos = text.find(first_word, start_pos)
                if pos == -1:
                    break
                
                # Проверяем, что после первого слова идут остальные
                remaining_text = text[pos:pos + len(header_text) + 50]
                if all(word in remaining_text[:100] for word in words[1:]):
                    return pos
                
                start_pos = pos + 1
        
        return None
    
    def _determine_visual_header_level(self, visual_header: Dict) -> int:
        """
        Определение уровня заголовка на основе визуальных характеристик
        
        Args:
            visual_header: Информация о визуальном заголовке
            
        Returns:
            Уровень заголовка (0 = основной, 1 = подраздел)
        """
        font_size = visual_header.get('font_size', 0)
        is_bold = visual_header.get('is_bold', False)
        score = visual_header.get('score', 0)
        text = visual_header.get('text', '')
        
        # Основные заголовки: большой шрифт + жирный
        if font_size > 14 and is_bold:
            return 0
        
        # Высокий score обычно означает важный заголовок
        if score > 3.0:
            return 0
        
        # Нумерованные заголовки верхнего уровня
        if re.match(r'^\d+[\.\s]', text) and not re.match(r'^\d+\.\d+', text):
            return 0
        
        # По умолчанию - подраздел
        return 1
    
    def _find_all_headers(self, text: str) -> List[Dict]:
        """
        Поиск всех заголовков в тексте
        
        Args:
            text: Текст статьи
            
        Returns:
            Список заголовков с позициями
        """
        headers = []
        
        # Паттерны для поиска заголовков (строгие критерии)
        header_patterns = [
            # 1. Нумерованные разделы (начинаются с цифры)
            r'\n\s*(\d+\.?\s+[A-Za-z][^.\n]{2,60})\s*\n',  # "1 Introduction", "2 Methods"
            r'\n\s*(\d+\.\d+\.?\s+[A-Za-z][^.\n]{2,60})\s*\n',  # "2.1 Data Collection"
            r'\n\s*(\d+\.\d+\.\d+\.?\s+[A-Za-z][^.\n]{2,60})\s*\n',  # "2.1.1 Details"
            
            # 2. Разделы начинающиеся с буквы + пробел (приложения)
            r'\n\s*([A-Z]\s+[A-Za-z][^.\n]{2,60})\s*\n',  # "A Additional Details", "B Experiments"
            
            # 3. Стандартные разделы статьи (точные совпадения)
            r'\n\s*(Abstract|ABSTRACT)\s*\n',
            r'\n\s*(Introduction|INTRODUCTION)\s*\n', 
            r'\n\s*(Related [Ww]ork|RELATED WORK)\s*\n',
            r'\n\s*(Background|BACKGROUND)\s*\n',
            r'\n\s*(Methods?|METHODS?|Methodology|METHODOLOGY)\s*\n',
            r'\n\s*(Results?|RESULTS?)\s*\n',
            r'\n\s*(Discussion|DISCUSSION)\s*\n',
            r'\n\s*(Conclusions?|CONCLUSIONS?)\s*\n',
            r'\n\s*(Acknowledgments?|ACKNOWLEDGMENTS?|Acknowledgements|ACKNOWLEDGEMENTS)\s*\n',
            r'\n\s*(References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*\n',
            r'\n\s*(Appendix|APPENDIX)\s*\n',
            r'\n\s*(Appendix [A-Z]|APPENDIX [A-Z])\s*\n',  # "Appendix A", "Appendix B"
        ]
        
        for pattern in header_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                title = match.group(1).strip()
                
                if self._is_valid_header(title):
                    headers.append({
                        'title': title,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'match_start': match.start(1),
                        'match_end': match.end(1),
                        'level': self._determine_header_level(title)
                    })
        
        # Сортируем заголовки по позиции
        headers.sort(key=lambda x: x['start_pos'])
        
        # Удаляем дубликаты (заголовки с одинаковыми названиями в близких позициях)
        headers = self._remove_duplicate_headers(headers)
        
        logger.info(f"Найдено {len(headers)} заголовков")
        return headers
    
    def _is_valid_header(self, title: str) -> bool:
        """
        Проверка валидности заголовка по строгим критериям:
        1. Начинается с цифры (1 Introduction, 2.1 Methods)
        2. Начинается с буквы + пробел (E Additional Details)
        3. Стандартные разделы (Abstract, Acknowledgments, References)
        
        Args:
            title: Заголовок для проверки
            
        Returns:
            True если заголовок валидный
        """
        title = title.strip()
        
        # Базовые фильтры
        if len(title) < 3 or len(title) > 100:
            return False
        
        # Исключаем URL и email
        if '@' in title or 'http' in title.lower():
            return False
        
        # Исключаем математические выражения
        math_symbols = ['=', '≤', '≥', '±', '∞', '∑', '∏', '∫']
        if any(symbol in title for symbol in math_symbols):
            return False
        
        # СТРОГИЕ КРИТЕРИИ:
        
        # 1. Начинается с цифры (нумерованные разделы)
        if re.match(r'^\d+', title):
            # Должен иметь текст после числа (с точкой или без)
            if re.match(r'^\d+(\.\d+)*[\.\s]*[A-Za-z]', title):
                return True
        
        # 2. Начинается с одной буквы + пробел (приложения)
        if re.match(r'^[A-Z]\s+[A-Za-z]', title):
            return True
        
        # 3. Стандартные разделы статьи
        standard_sections = [
            'Abstract', 'ABSTRACT',
            'Introduction', 'INTRODUCTION', 
            'Related Work', 'RELATED WORK',
            'Background', 'BACKGROUND',
            'Methods', 'METHODS', 'Methodology', 'METHODOLOGY',
            'Results', 'RESULTS',
            'Discussion', 'DISCUSSION',
            'Conclusion', 'CONCLUSION', 'Conclusions', 'CONCLUSIONS',
            'Acknowledgments', 'ACKNOWLEDGMENTS', 'Acknowledgements', 'ACKNOWLEDGEMENTS',
            'References', 'REFERENCES',
            'Bibliography', 'BIBLIOGRAPHY',
            'Appendix', 'APPENDIX'
        ]
        
        # Точное совпадение со стандартными разделами
        if title in standard_sections:
            return True
        
        # Appendix с буквой (Appendix A, Appendix B)
        if re.match(r'^Appendix\s+[A-Z]', title, re.IGNORECASE):
            return True
        
        # Все остальное отклоняем
        return False
    
    def _determine_header_level(self, title: str) -> int:
        """
        Определение уровня заголовка
        
        Args:
            title: Заголовок
            
        Returns:
            Уровень заголовка (0 = основной, 1 = подраздел, и т.д.)
        """
        # Основные разделы статьи
        main_sections = [
            'Abstract', 'Introduction', 'Related work', 'Methods', 'Methodology',
            'Results', 'Discussion', 'Conclusion', 'Conclusions', 'References'
        ]
        
        if any(main in title for main in main_sections):
            return 0
        
        # Нумерованные разделы
        if re.match(r'^\d+\s+', title):  # "1 Introduction"
            return 0
        
        if re.match(r'^\d+\.\d+\s+', title):  # "2.1 Methods"
            return 1
        
        if re.match(r'^\d+\.\d+\.\d+\s+', title):  # "2.1.1 Details"
            return 2
        
        # По умолчанию
        return 1
    
    def _remove_duplicate_headers(self, headers: List[Dict]) -> List[Dict]:
        """
        Удаление дублирующихся заголовков
        
        Args:
            headers: Список заголовков
            
        Returns:
            Список без дубликатов
        """
        if not headers:
            return headers
        
        unique_headers = []
        prev_header = None
        
        for header in headers:
            # Проверяем на дубликаты по названию и близкой позиции
            if prev_header and header['title'] == prev_header['title']:
                if abs(header['start_pos'] - prev_header['start_pos']) < 100:
                    continue  # Пропускаем дубликат
            
            unique_headers.append(header)
            prev_header = header
        
        return unique_headers
    
    def _create_sections_from_headers(self, text: str, headers: List[Dict]) -> List[Dict]:
        """
        Создание разделов на основе найденных заголовков
        
        Args:
            text: Полный текст
            headers: Список заголовков
            
        Returns:
            Список разделов
        """
        sections = []
        
        # 1. Создаем секцию "Title" для текста до первой секции
        if headers and headers[0]['start_pos'] > 0:
            title_content = text[:headers[0]['start_pos']].strip()
            if title_content:
                sections.append({
                    'title': 'Title',
                    'content': title_content,
                    'start_pos': 0,
                    'end_pos': headers[0]['start_pos'],
                    'level': 0
                })
        
        # 2. Создаем секции для каждого заголовка
        for i, header in enumerate(headers):
            # Начало секции = позиция заголовка
            section_start = header['start_pos']
            
            # Конец секции = начало следующего заголовка или конец текста
            if i + 1 < len(headers):
                section_end = headers[i + 1]['start_pos']
            else:
                section_end = len(text)
            
            # Извлекаем содержимое секции (включая заголовок)
            section_content = text[section_start:section_end].strip()
            
            # Убираем лишние пробелы и переносы
            section_content = re.sub(r'\n\s*\n\s*\n', '\n\n', section_content)
            
            if section_content:
                sections.append({
                    'title': header['title'],
                    'content': section_content,
                    'start_pos': section_start,
                    'end_pos': section_end,
                    'level': header['level']
                })
        
        return sections
    
    def _create_chunk(self, text: str, section_title: str, chunk_index: int, 
                     arxiv_id: str, section_start_pos: int = 0, 
                     page_info: Optional[Dict] = None) -> Dict:
        """
        Создание отдельного чанка с метаданными
        
        Args:
            text: Текст чанка
            section_title: Название раздела
            chunk_index: Индекс чанка
            arxiv_id: ID статьи
            section_start_pos: Начальная позиция раздела в тексте
            page_info: Информация о страницах
            
        Returns:
            Чанк с метаданными
        """
        return {
            'text': text,
            'metadata': {
                'section': section_title,
                'chunk_index': chunk_index,
                'arxiv_id': arxiv_id,
                'chunk_type': 'section_based',
                'section_start_pos': section_start_pos
            },
            'chunk_id': f"{arxiv_id}_{chunk_index}"
        }
    
    def _split_large_section(self, section_content: str, section_title: str,
                           start_chunk_index: int, arxiv_id: str, 
                           section_start_pos: int = 0,
                           page_info: Optional[Dict] = None) -> List[Dict]:
        """
        Разбиение большого раздела на подчанки
        
        Args:
            section_content: Содержимое раздела
            section_title: Название раздела
            start_chunk_index: Начальный индекс чанка
            arxiv_id: ID статьи
            section_start_pos: Позиция начала раздела
            page_info: Информация о страницах
            
        Returns:
            Список подчанков
        """
        if not self.text_splitter:
            # Простое разбиение по размеру
            chunks = []
            chunk_size = self.max_section_size
            
            for i in range(0, len(section_content), chunk_size - self.chunk_overlap):
                chunk_text = section_content[i:i + chunk_size]
                if chunk_text.strip():
                    chunk = self._create_chunk(
                        text=chunk_text,
                        section_title=section_title,
                        chunk_index=start_chunk_index + len(chunks),
                        arxiv_id=arxiv_id,
                        section_start_pos=section_start_pos + i,
                        page_info=page_info
                    )
                    chunks.append(chunk)
            
            return chunks
        
        # Используем RecursiveCharacterTextSplitter для больших разделов
        sub_texts = self.text_splitter.split_text(section_content)
        
        chunks = []
        for i, sub_text in enumerate(sub_texts):
            if sub_text.strip():
                chunk = self._create_chunk(
                    text=sub_text,
                    section_title=section_title,
                    chunk_index=start_chunk_index + i,
                    arxiv_id=arxiv_id,
                    section_start_pos=section_start_pos,
                    page_info=page_info
                )
                chunks.append(chunk)
        
        return chunks

# Глобальный экземпляр чанкера
section_chunker = SectionBasedChunker()
