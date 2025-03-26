"""
File operation tools for the book_creator package.

This module provides tools for reading, writing, and listing files and directories.
"""

import os
import logging
from typing import List
from smolagents import tool

logger = logging.getLogger(__name__)

@tool
def save_text(text: str, file_path: str) -> str:
    """
    Saves the provided text to a file at the specified path.
    
    Args:
        text: The text to save to the file.
        file_path: The path to save the text to.
        
    Returns:
        str: Confirmation message with the file path.
    """
    with open(file_path, 'w') as f:
        f.write(text)
    return f"Successfully saved text to {file_path}"

@tool
def create_folder(folder_path: str) -> str:
    """
    Creates a folder at the specified path.

    Args:
        folder_path: The path to create the folder at.
        
    Returns:
        str: Confirmation message with the folder path.
    """
    os.makedirs(folder_path, exist_ok=True)
    return f"Successfully created folder at {folder_path}"   

@tool
def read_text_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Reads and returns the content of a text file with flexible encoding handling.
    
    Args:
        file_path: The path to the text file to read.
        encoding: The encoding to use when reading the file. Defaults to 'utf-8'.
        
    Returns:
        str: The content of the text file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        IOError: If there's an error reading the file.
    """
    encodings_to_try = [encoding, 'latin-1', 'cp1252', 'iso-8859-1', 'utf-8-sig']
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            logger.info(f"Successfully read file {file_path} with encoding {enc}")
            return content
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode {file_path} with encoding {enc}, trying next encoding")
            continue
    
    try:
        with open(file_path, 'rb') as f:
            binary_content = f.read()
        content = binary_content.decode('utf-8', errors='replace')
        logger.warning(f"Read {file_path} with replacement of invalid characters")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

@tool
def read_image_file(file_path: str) -> str:
    """
    Reads an image file and returns its path if it exists.
    
    Args:
        file_path: The path to the image file to verify.
        
    Returns:
        str: The verified path to the image file.
        
    Raises:
        FileNotFoundError: If the image file doesn't exist.
    """
    if not os.path.exists(file_path):
        logger.error(f"Image file not found: {file_path}")
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    valid_extensions = ['.png', '.jpg', '.jpeg']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        logger.warning(f"File may not be an image: {file_path}")
    
    return file_path

@tool
def list_directory(directory_path: str = ".", recursive: bool = True, indent: int = 0) -> List[str]:
    """
    Lists files and directories in the specified directory.
    
    Args:
        directory_path: The path to the directory to list. Defaults to current directory.
        recursive: Whether to recursively list subdirectories. Defaults to False.
        indent: Indentation level for recursive listing. Used internally.
        
    Returns:
        List[str]: A list of file and directory paths in the specified directory.
        
    Raises:
        FileNotFoundError: If the directory doesn't exist.
        PermissionError: If there's no permission to access the directory.
    """
    try:
        cwd = os.getcwd()
        logger.info(f"Current working directory: {cwd}")
        
        contents = os.listdir(directory_path)
        
        contents.sort(key=lambda x: (0 if os.path.isdir(os.path.join(directory_path, x)) else 1, x))
        
        result = []
        
        for item in contents:
            full_path = os.path.join(directory_path, item)
            is_dir = os.path.isdir(full_path)
            
            result.append(full_path)
            
            if recursive and is_dir:
                try:
                    subdirectory_contents = list_directory(full_path, recursive=True, indent=indent+1)
                    result.extend(subdirectory_contents)
                except (PermissionError, FileNotFoundError) as e:
                    logger.warning(f"Could not access subdirectory {full_path}: {e}")
        
        return result
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied to access directory: {directory_path}")
        raise

@tool
def list_directory_tree(directory_path: str = ".") -> str:
    """
    Generates a formatted tree representation of the directory structure.
    
    Args:
        directory_path: The path to the directory to list. Defaults to current directory.
        
    Returns:
        str: A formatted string showing the directory tree structure.
        
    Raises:
        FileNotFoundError: If the directory doesn't exist.
        PermissionError: If there's no permission to access the directory.
    """
    def _generate_tree(dir_path: str, prefix: str = "") -> List[str]:
        """
        Helper function to recursively generate tree structure.
        
        Args:
            dir_path: Current directory path.
            prefix: Prefix for the current line (for formatting).
            
        Returns:
            List[str]: Lines of the tree structure.
        """
        if not os.path.isdir(dir_path):
            return [f"{prefix}├── {os.path.basename(dir_path)}"]
            
        result = [f"{prefix}├── {os.path.basename(dir_path)}/"]
        
        try:
            items = os.listdir(dir_path)
            items.sort(key=lambda x: (0 if os.path.isdir(os.path.join(dir_path, x)) else 1, x))
            
            for i, item in enumerate(items):
                is_last = (i == len(items) - 1)
                item_path = os.path.join(dir_path, item)
                
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                if os.path.isdir(item_path):
                    subtree = _generate_tree(item_path, next_prefix)
                    result.extend(subtree)
                else:
                    result.append(f"{prefix}{'└── ' if is_last else '├── '}{item}")
                    
            return result
        except (PermissionError, FileNotFoundError) as e:
            logger.warning(f"Could not access {dir_path}: {e}")
            return [f"{prefix}├── {os.path.basename(dir_path)}/ (access denied)"]
    
    try:
        logger.info(f"Generating directory tree for: {directory_path}")
        
        tree_lines = [os.path.abspath(directory_path)]
        
        for item in os.listdir(directory_path):
            is_last = (item == os.listdir(directory_path)[-1])
            item_path = os.path.join(directory_path, item)
            
            if os.path.isdir(item_path):
                subtree = _generate_tree(item_path, "    " if is_last else "│   ")
                tree_lines.extend(subtree)
            else:
                tree_lines.append(f"{'└── ' if is_last else '├── '}{item}")
        
        return "\n".join(tree_lines)
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied to access directory: {directory_path}")
        raise