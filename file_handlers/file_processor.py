import os
import re
from typing import Optional


class FileProcessor:
    """Process various file formats and extract text content"""

    @staticmethod
    def extract_text_from_file(file_path: str) -> Optional[str]:
        """Extract text from various file formats"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.txt':
            return FileProcessor._read_text_file(file_path)
        elif file_extension in ['.md', '.markdown']:
            return FileProcessor._read_markdown_file(file_path)
        elif file_extension == '.tex':
            return FileProcessor._read_latex_file(file_path)
        else:
            # For now, only support text-based files
            # PDF and DOCX support will be added when dependencies are available
            raise ValueError(
                f"File format not yet supported: {file_extension}. Please use text files (.txt), markdown (.md), or LaTeX (.tex) files."
            )

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        """Read plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def _read_markdown_file(file_path: str) -> str:
        """Read markdown file and return plain text"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Remove markdown formatting (basic cleanup)
        content = re.sub(r'#+\s*', '', content)  # Remove headers
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*([^*]+)\*', r'\1', content)  # Remove italic
        content = re.sub(r'`([^`]+)`', r'\1', content)  # Remove inline code
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1',
                         content)  # Remove links
        return content.strip()

    @staticmethod
    def _read_latex_file(file_path: str) -> str:
        """Extract text from LaTeX file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Remove LaTeX commands and comments
        content = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^}]*)\}', r'\2',
                         content)
        content = re.sub(r'%.*$', '', content,
                         flags=re.MULTILINE)  # Remove comments
        content = re.sub(r'\\[a-zA-Z]+', '',
                         content)  # Remove remaining commands
        content = re.sub(r'\\\\(\[.*?\])?', '',
                         content)  # Remove line breaks with options
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace

        return content.strip()

    @staticmethod
    def process_input(input_data: str) -> str:
        """Process input text (clean and normalize)"""
        if not input_data or not isinstance(input_data, str):
            return ""

        # Basic text cleaning
        text = input_data.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '',
                      text)  # Remove control characters

        return text

    @staticmethod
    def is_valid_text(text: str, min_length: int = 50) -> bool:
        """Check if text is valid for analysis"""
        if not text or not isinstance(text, str):
            return False

        # Remove whitespace and check length
        clean_text = re.sub(r'\s+', '', text)
        return len(clean_text) >= min_length
