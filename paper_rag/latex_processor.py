"""
Модуль для обработки LaTeX файлов
"""

import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class LatexProcessor:
    """
    Класс для обработки LaTeX файлов и извлечения структурированного текста
    """
    
    def __init__(self):
        self.section_patterns = [
            r'\\section\*?\{([^}]+)\}',  # \section{Title} или \section*{Title}
            r'\\subsection\*?\{([^}]+)\}',  # \subsection{Title}
            r'\\subsubsection\*?\{([^}]+)\}',  # \subsubsection{Title}
            r'\\chapter\*?\{([^}]+)\}',  # \chapter{Title}
            r'\\part\*?\{([^}]+)\}',  # \part{Title}
        ]
        
        self.environment_patterns = [
            r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}',  # \begin{env}...\end{env}
            r'\\begin\{([^}]+)\}',  # \begin{env}
            r'\\end\{([^}]+)\}',  # \end{env}
        ]
        
        self.command_patterns = [
            r'\\title\{([^}]+)\}',  # \title{Title}
            r'\\author\{([^}]+)\}',  # \author{Author}
            r'\\abstract\*?\{([^}]+)\}',  # \abstract{Abstract}
            r'\\maketitle',  # \maketitle
            r'\\tableofcontents',  # \tableofcontents
        ]
    
    def extract_from_source(self, source_path: str) -> Optional[Dict]:
        """
        Извлечение текста из исходного кода статьи
        
        Args:
            source_path: Путь к архиву с исходным кодом
            
        Returns:
            Словарь с извлеченным текстом и структурой
        """
        try:
            # Проверяем реальный тип файла, а не только расширение
            file_type = self._detect_file_type(source_path)
            
            if file_type == 'pdf':
                logger.warning(f"Файл {source_path} является PDF, а не LaTeX архивом")
                return None
            elif file_type != 'gzip':
                logger.error(f"Неподдерживаемый тип файла: {file_type} для {source_path}")
                return None
            
            # Создаем временную папку для распаковки
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Распаковываем архив
                with tarfile.open(source_path, 'r:gz') as tar:
                    tar.extractall(temp_path)
                
                # Ищем основной LaTeX файл
                main_tex = self._find_main_tex(temp_path)
                if not main_tex:
                    logger.error("Не найден основной LaTeX файл")
                    return None
                
                # Обрабатываем LaTeX файл
                return self._process_latex_file(main_tex)
                
        except Exception as e:
            logger.error(f"Ошибка при обработке исходного кода: {e}")
            return None
    
    def _detect_file_type(self, file_path: str) -> str:
        """
        Определение реального типа файла по магическим числам
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Тип файла: 'gzip', 'pdf', 'zip', 'unknown'
        """
        try:
            with open(file_path, 'rb') as f:
                # Читаем первые 10 байт для определения типа
                header = f.read(10)
                
                # Проверяем магические числа
                if header.startswith(b'\x1f\x8b'):
                    return 'gzip'  # gzip архив
                elif header.startswith(b'%PDF'):
                    return 'pdf'    # PDF документ
                elif header.startswith(b'PK'):
                    return 'zip'    # ZIP архив
                else:
                    # Дополнительные проверки
                    if header.startswith(b'\\documentclass') or header.startswith(b'\\begin{document'):
                        return 'latex'  # LaTeX файл
                    elif header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                        return 'html'   # HTML файл
                    else:
                        return 'unknown'
                        
        except Exception as e:
            logger.error(f"Ошибка при определении типа файла {file_path}: {e}")
            return 'unknown'
    
    def _find_main_tex(self, temp_path: Path) -> Optional[Path]:
        """
        Поиск основного LaTeX файла
        
        Args:
            temp_path: Путь к временной папке
            
        Returns:
            Путь к основному .tex файлу
        """
        # Ищем .tex файлы
        tex_files = list(temp_path.rglob("*.tex"))
        
        if not tex_files:
            return None
        
        # Приоритет: main.tex, article.tex, paper.tex, первый найденный
        priority_names = ['main.tex', 'article.tex', 'paper.tex']
        
        for name in priority_names:
            for tex_file in tex_files:
                if tex_file.name.lower() == name:
                    return tex_file
        
        # Возвращаем первый найденный
        return tex_files[0]
    
    def _process_latex_file(self, tex_path: Path) -> Dict:
        """
        Обработка LaTeX файла
        
        Args:
            tex_path: Путь к .tex файлу
            
        Returns:
            Словарь с извлеченным текстом и структурой
        """
        try:
            with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Извлекаем структуру
            structure = self._extract_structure(content)
            
            # Извлекаем чистый текст
            clean_text = self._extract_clean_text(content)
            
            # Разбиваем на секции
            sections = self._split_into_sections(content, structure)
            
            return {
                'text': clean_text,
                'structure': structure,
                'sections': sections,
                'metadata': {
                    'source': 'latex',
                    'file_path': str(tex_path),
                    'extraction_method': 'latex_processor',
                    'total_sections': len(sections),
                    'file_size': len(content)
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка при обработке LaTeX файла: {e}")
            return None
    
    def _extract_structure(self, content: str) -> Dict:
        """
        Извлечение структуры документа
        
        Args:
            content: Содержимое LaTeX файла
            
        Returns:
            Словарь со структурой
        """
        structure = {
            'title': None,
            'authors': [],
            'abstract': None,
            'sections': [],
            'environments': [],
            'commands': []
        }
        
        # Извлекаем заголовок
        title_match = re.search(r'\\title\{([^}]+)\}', content)
        if title_match:
            structure['title'] = title_match.group(1).strip()
        
        # Извлекаем авторов
        author_matches = re.findall(r'\\author\{([^}]+)\}', content)
        for match in author_matches:
            # Разбиваем авторов по \and или ,
            authors = re.split(r'\\and|,', match)
            structure['authors'].extend([a.strip() for a in authors if a.strip()])
        
        # Извлекаем аннотацию
        abstract_match = re.search(r'\\abstract\*?\{([^}]+)\}', content)
        if abstract_match:
            structure['abstract'] = abstract_match.group(1).strip()
        
        # Извлекаем секции
        for pattern in self.section_patterns:
            # Используем finditer для получения полного совпадения и позиции
            for match in re.finditer(pattern, content):
                section_title = match.group(1).strip()  # Содержимое в скобках
                section_type = pattern.split('\\')[1].split('{')[0]  # Тип секции
                
                structure['sections'].append({
                    'type': section_type,
                    'title': section_title,
                    'level': self._get_section_level(pattern),
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # Извлекаем окружения
        for pattern in self.environment_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    env_name = match[0]
                    env_content = match[1] if len(match) > 1 else ""
                else:
                    env_name = match
                    env_content = ""
                
                structure['environments'].append({
                    'name': env_name,
                    'content': env_content[:200] + "..." if len(env_content) > 200 else env_content
                })
        
        return structure
    
    def _get_section_level(self, pattern: str) -> int:
        """Определяет уровень секции"""
        if 'chapter' in pattern or 'part' in pattern:
            return 1
        elif 'section' in pattern:
            return 2
        elif 'subsection' in pattern:
            return 3
        elif 'subsubsection' in pattern:
            return 4
        return 1
    
    def _extract_clean_text(self, content: str) -> str:
        """
        Извлечение чистого текста из LaTeX
        
        Args:
            content: Содержимое LaTeX файла
            
        Returns:
            Очищенный текст
        """
        # Убираем LaTeX команды
        text = content
        
        # Убираем комментарии
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
        
        # Убираем LaTeX команды с параметрами
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        
        # Убираем простые LaTeX команды
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Убираем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _split_into_sections(self, content: str, structure: Dict) -> List[Dict]:
        """
        Разбиение текста на секции
        
        Args:
            content: Содержимое LaTeX файла
            structure: Структура документа
            
        Returns:
            Список секций с текстом
        """
        sections = []
        
        # Сортируем секции по позиции в тексте
        section_positions = []
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, content):
                section_positions.append({
                    'match': match,
                    'type': pattern.split('\\')[1].split('{')[0],
                    'title': match.group(1).strip(),
                    'start': match.start(),
                    'level': self._get_section_level(pattern)
                })
        
        # Сортируем по позиции
        section_positions.sort(key=lambda x: x['start'])
        
        # Создаем секцию "Title" для текста до первой секции
        if section_positions and section_positions[0]['start'] > 0:
            title_text = content[:section_positions[0]['start']]
            clean_title_text = self._extract_clean_text(title_text)
            if clean_title_text.strip():
                sections.append({
                    'type': 'title',
                    'title': 'Title',
                    'level': 0,
                    'text': clean_title_text,
                    'start_pos': 0,
                    'end_pos': section_positions[0]['start'],
                    'char_count': len(clean_title_text),
                    'word_count': len(clean_title_text.split())
                })
        
        # Разбиваем на секции
        for i, section_info in enumerate(section_positions):
            start_pos = section_info['start']
            end_pos = section_positions[i + 1]['start'] if i + 1 < len(section_positions) else len(content)
            
            section_text = content[start_pos:end_pos]
            
            # Очищаем текст секции
            clean_section_text = self._extract_clean_text(section_text)
            
            sections.append({
                'type': section_info['type'],
                'title': section_info['title'],
                'level': section_info['level'],
                'text': clean_section_text,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'char_count': len(clean_section_text),
                'word_count': len(clean_section_text.split())
            })
        
        return sections
    
    def get_section_hierarchy(self, sections: List[Dict]) -> List[Dict]:
        """
        Строит иерархию секций
        
        Args:
            sections: Список секций
            
        Returns:
            Иерархическая структура секций
        """
        hierarchy = []
        current_level_1 = None
        current_level_2 = None
        
        for section in sections:
            if section['level'] == 1:  # chapter/part
                current_level_1 = {
                    'title': section['title'],
                    'type': section['type'],
                    'text': section['text'],
                    'subsections': []
                }
                hierarchy.append(current_level_1)
                current_level_2 = None
                
            elif section['level'] == 2:  # section
                if current_level_1:
                    current_level_2 = {
                        'title': section['title'],
                        'type': section['type'],
                        'text': section['text'],
                        'subsections': []
                    }
                    current_level_1['subsections'].append(current_level_2)
                else:
                    # Создаем корневую секцию
                    current_level_2 = {
                        'title': section['title'],
                        'type': section['type'],
                        'text': section['text'],
                        'subsections': []
                    }
                    hierarchy.append(current_level_2)
                    
            elif section['level'] >= 3:  # subsection и ниже
                if current_level_2:
                    subsection = {
                        'title': section['title'],
                        'type': section['type'],
                        'text': section['text'],
                        'level': section['level']
                    }
                    current_level_2['subsections'].append(subsection)
        
        return hierarchy
