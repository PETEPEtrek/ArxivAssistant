"""
Модуль для нарезки текста на чанки с метаданными
"""

import re
from typing import List, Dict, Optional
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .section_chunking import section_chunker

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Класс для разбиения текста на чанки с сохранением метаданных
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Инициализация чанкера
        
        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие между чанками
            separators: Разделители для текста
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            # Приоритетные разделители для научных текстов
            self.separators = [
                "\n\n\n",  # Разделы
                "\n\n",    # Параграфы
                "\n",      # Строки
                ". ",      # Предложения
                "! ",
                "? ",
                "; ",
                ", ",
                " ",       # Слова
                ""         # Символы
            ]
        else:
            self.separators = separators
        
        self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                keep_separator=True
            )
    
    def chunk_text(self, extracted_data: Dict) -> List[Dict]:
        """
        Разбиение извлеченного текста на чанки
        
        Args:
            extracted_data: Данные из pdf_processor
            
        Returns:
            Список чанков с метаданными
        """
        if not extracted_data or not extracted_data.get('text'):
            logger.error("Нет текста для разбиения на чанки")
            return []
        
        text = extracted_data['text']
        metadata = extracted_data.get('metadata', {})
        
        # Используем новый метод section-based чанкинга
        return self.chunk_text_by_sections(extracted_data)
    
    def chunk_text_by_sections(self, extracted_data: Dict) -> List[Dict]:
        """
        Разбиение текста на чанки по разделам (новый метод)
        
        Args:
            extracted_data: Данные из pdf_processor
            
        Returns:
            Список чанков с метаданными
        """
        if not extracted_data or not extracted_data.get('text'):
            logger.error("Нет текста для разбиения на чанки")
            return []
        
        text = extracted_data['text']
        metadata = extracted_data.get('metadata', {})
        arxiv_id = metadata.get('arxiv_id', 'unknown')
        
        logger.info(f"Начало section-based чанкинга для статьи {arxiv_id}")
        
        try:
            # Извлекаем разделы с помощью нового чанкера
            pdf_path = metadata.get('pdf_path')  # Путь к PDF для визуального анализа
            sections = section_chunker.extract_sections(text, pdf_path)
            
            # Создаем чанки из разделов
            chunks = section_chunker.chunk_sections(
                sections=sections,
                arxiv_id=arxiv_id,
                page_info=metadata.get('page_info')
            )
            
            logger.info(f"Section-based чанкинг завершен: {len(chunks)} чанков из {len(sections)} разделов")
            return chunks
            
        except Exception as e:
            logger.error(f"Ошибка section-based чанкинга: {e}")
            # Fallback к старому методу
            logger.info("Переход к старому методу чанкинга")
            return self.chunk_text_legacy(extracted_data)
    
    def chunk_text_legacy(self, extracted_data: Dict) -> List[Dict]:
        """
        Старый метод разбиения текста на чанки (для fallback)
        
        Args:
            extracted_data: Данные из pdf_processor
            
        Returns:
            Список чанков с метаданными
        """
        if not extracted_data or not extracted_data.get('text'):
            logger.error("Нет текста для разбиения на чанки")
            return []
        
        text = extracted_data['text']
        metadata = extracted_data.get('metadata', {})
        
        # Определяем разделы если доступны
        sections = self._extract_sections(text)
        
        # Разбиваем на чанки
        if self.text_splitter:
            chunks = self._chunk_with_langchain(text, sections, metadata)
        else:
            chunks = self._chunk_simple(text, sections, metadata)
        
        logger.info(f"Создано {len(chunks)} чанков из текста длиной {len(text)} символов")
        return chunks
    
    def _extract_sections(self, text: str) -> List[Dict]:
        """
        Извлечение разделов из текста
        
        Args:
            text: Полный текст статьи
            extracted_data: Исходные данные с возможными разделами
            
        Returns:
            Список разделов с позициями
        """
        sections = []
        
        # Ищем разделы по паттернам (улучшенные для научных статей)
        section_patterns = [
            # Основные разделы (точные названия)
            r'\n\s*(Abstract|ABSTRACT)\s*\n',
            r'\n\s*(Introduction|INTRODUCTION)\s*\n',
            r'\n\s*(Related [Ww]ork|RELATED WORK)\s*\n',
            r'\n\s*(Background|BACKGROUND)\s*\n',
            r'\n\s*(Methods?|METHODS?|Methodology|METHODOLOGY)\s*\n',
            r'\n\s*(Experiments?|EXPERIMENTS?|Experimental Setup)\s*\n',
            r'\n\s*(Results?|RESULTS?)\s*\n',
            r'\n\s*(Discussion|DISCUSSION)\s*\n',
            r'\n\s*(Conclusions?|CONCLUSIONS?)\s*\n',
            r'\n\s*(Future [Ww]ork|FUTURE WORK)\s*\n',
            r'\n\s*(Acknowledgments?|ACKNOWLEDGMENTS?)\s*\n',
            r'\n\s*(References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*\n',
            r'\n\s*(Appendix [A-Z]?|APPENDIX [A-Z]?)\s*\n',
            
            # Нумерованные разделы (строгие паттерны)
            r'\n\s*(\d+\.?\s+[A-Z][a-z][^.\n]{3,40})\s*\n',  # "1 Introduction", "2.1 Methods"
            r'\n\s*(\d+\.?\d*\.?\s+[A-Z][a-z][^.\n]{3,40})\s*\n',  # "2.1 Data Collection"
            
            # Case studies и специальные разделы
            r'\n\s*(Case Study #?\d+[^.\n]{0,40})\s*\n',
            r'\n\s*(Problem Setup|PROBLEM SETUP)\s*\n',
            r'\n\s*(Evaluation|EVALUATION)\s*\n',
            r'\n\s*(Analysis|ANALYSIS)\s*\n',
            
            # Заголовки полностью заглавными (только если они короткие и осмысленные)
            r'\n\s*([A-Z][A-Z\s]{4,25})\s*\n(?=[A-Z][a-z])',  # За заголовком должен идти текст
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                title = match.group(1).strip()
                
                # Фильтруем нежелательные "секции"
                if self._is_valid_section_title(title):
                    sections.append({
                        'title': title,
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    })
        
        # Сортируем по позиции
        sections.sort(key=lambda x: x['start_pos'])
        
        # Добавляем границы разделов
        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                section['text_end'] = sections[i + 1]['start_pos']
            else:
                section['text_end'] = len(text)
        
        return sections
    
    def _chunk_with_langchain(self, text: str, sections: List[Dict], metadata: Dict) -> List[Dict]:
        """
        Разбиение текста с помощью langchain
        
        Args:
            text: Текст для разбиения
            sections: Информация о разделах
            metadata: Метаданные исходного документа
            
        Returns:
            Список чанков
        """
        try:
            # Разбиваем текст на чанки
            text_chunks = self.text_splitter.split_text(text)
            
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                # Определяем раздел для чанка
                section_info = self._find_section_for_chunk(chunk_text, text, sections)
                
                chunk = {
                    'text': chunk_text.strip(),
                    'chunk_id': i,
                    'metadata': {
                        **metadata,
                        'chunk_size': len(chunk_text),
                        'section': section_info['title'] if section_info else 'Unknown',
                        'section_number': section_info['number'] if section_info else None,
                        'chunk_index': i
                    }
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Ошибка при разбиении с langchain: {e}")
            return self._chunk_simple(text, sections, metadata)
    
    def _chunk_simple(self, text: str, sections: List[Dict], metadata: Dict) -> List[Dict]:
        """
        Простое разбиение текста на чанки
        
        Args:
            text: Текст для разбиения
            sections: Информация о разделах  
            metadata: Метаданные исходного документа
            
        Returns:
            Список чанков
        """
        chunks = []
        chunk_start = 0
        chunk_id = 0
        
        while chunk_start < len(text):
            # Определяем конец чанка
            chunk_end = min(chunk_start + self.chunk_size, len(text))
            
            # Пытаемся найти хорошее место для разрыва
            if chunk_end < len(text):
                # Ищем ближайший разделитель
                for sep in self.separators:
                    if not sep:
                        continue
                    
                    # Ищем разделитель в последних 200 символах чанка
                    search_start = max(chunk_end - 200, chunk_start)
                    sep_pos = text.rfind(sep, search_start, chunk_end)
                    
                    if sep_pos > chunk_start:
                        chunk_end = sep_pos + len(sep)
                        break
            
            # Извлекаем текст чанка
            chunk_text = text[chunk_start:chunk_end].strip()
            
            if len(chunk_text) < 50:  # Пропускаем слишком короткие чанки
                chunk_start = chunk_end - self.chunk_overlap
                continue
            
            # Определяем раздел
            section_info = self._find_section_for_chunk(chunk_text, text, sections)
            
            chunk = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                'metadata': {
                    **metadata,
                    'chunk_size': len(chunk_text),
                    'section': section_info['title'] if section_info else 'Unknown',
                    'section_number': section_info['number'] if section_info else None,
                    'chunk_index': chunk_id,
                    'start_pos': chunk_start,
                    'end_pos': chunk_end
                }
            }
            chunks.append(chunk)
            
            chunk_id += 1
            chunk_start = chunk_end - self.chunk_overlap
        
        return chunks
    
    def _find_section_for_chunk(self, chunk_text: str, full_text: str, sections: List[Dict]) -> Optional[Dict]:
        """
        Определение раздела для чанка
        
        Args:
            chunk_text: Текст чанка
            full_text: Полный текст документа
            sections: Список разделов
            
        Returns:
            Информация о разделе или None
        """
        if not sections:
            return None
        
        # Сначала проверяем, содержит ли сам чанк заголовок секции
        chunk_section = self._find_section_title_in_chunk(chunk_text, sections)
        if chunk_section:
            return chunk_section
        
        # Находим позицию чанка в полном тексте
        chunk_pos = full_text.find(chunk_text[:100])  # Используем первые 100 символов
        
        if chunk_pos == -1:
            return None
        
        # Находим подходящий раздел по позиции
        for i, section in enumerate(sections):
            if (section['start_pos'] <= chunk_pos <= section.get('text_end', len(full_text))):
                return {
                    'title': section['title'],
                    'number': i + 1
                }
        
        return None
    

    
    def _is_valid_section_title(self, title: str) -> bool:
        """
        Проверка валидности заголовка секции
        
        Args:
            title: Заголовок для проверки
            
        Returns:
            True если заголовок валидный
        """
        # Исключаем слишком длинные строки (вероятно, фрагменты текста)
        if len(title) > 60:
            return False
        
        # Исключаем строки с годами публикации (цитаты)
        if re.search(r'\b(19|20)\d{2}\b', title):
            return False
        
        # Исключаем строки с SQL кодом
        if re.search(r'\b(SELECT|FROM|WHERE|TABLE|ATTRIBUTE)\b', title, re.IGNORECASE):
            return False
        
        # Исключаем строки с множественными числами (данные таблиц)
        if len(re.findall(r'\d+', title)) > 2:
            return False
        
        # Исключаем строки, начинающиеся с маленькой буквы (фрагменты предложений)
        if title and title[0].islower():
            return False
        
        # Исключаем строки с математическими символами и формулами
        if re.search(r'[=<>±×÷∑∫πα-ωΑ-Ω]', title):
            return False
        
        # Исключаем общие фразы/фрагменты
        invalid_fragments = [
            'to study', 'in order to', 'we propose', 'we present', 'our approach',
            'the results', 'as shown in', 'figure', 'table', 'equation',
            'can be', 'should be', 'will be', 'has been', 'have been'
        ]
        
        title_lower = title.lower()
        for fragment in invalid_fragments:
            if fragment in title_lower:
                return False
        
        return True
    
    def _find_section_title_in_chunk(self, chunk_text: str, sections: List[Dict]) -> Optional[Dict]:
        """
        Поиск заголовка секции внутри самого чанка
        
        Args:
            chunk_text: Текст чанка
            sections: Список найденных секций
            
        Returns:
            Информация о секции, если заголовок найден в чанке
        """
        # Проверяем, содержит ли чанк заголовок секции
        for i, section in enumerate(sections):
            section_title = section['title']
            
            # Простая проверка на вхождение
            if section_title in chunk_text:
                return {
                    'title': section_title,
                    'number': i + 1
                }
            
            # Более сложная проверка для нумерованных секций
            # Например, "4 Case Study #1: Selective Text-to-SQL"
            if re.search(rf'\b{re.escape(section_title)}\b', chunk_text, re.IGNORECASE):
                return {
                    'title': section_title,
                    'number': i + 1
                }
            
            # Проверка для частичных совпадений нумерованных секций
            if section_title.startswith(('1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ')):
                # Извлекаем номер и название
                parts = section_title.split(' ', 1)
                if len(parts) >= 2:
                    section_number = parts[0]
                    section_name = parts[1]
                    
                    # Проверяем наличие номера секции в начале строки чанка
                    pattern = rf'^\s*{re.escape(section_number)}\s+{re.escape(section_name[:20])}'
                    if re.search(pattern, chunk_text, re.MULTILINE | re.IGNORECASE):
                        return {
                            'title': section_title,
                            'number': i + 1
                        }
        
        return None

# Глобальный экземпляр чанкера
text_chunker = TextChunker()
