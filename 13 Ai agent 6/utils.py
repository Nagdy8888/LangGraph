"""
Utility functions and helpers for the Multi-Modal Agent.

This module contains various utility functions that support the agent's
functionality, including file operations, data processing, and helpers.
"""

import os
import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import re

from config import WORKSPACE_DIR, SUPPORTED_FILE_TYPES, MAX_FILE_SIZE


def ensure_directory_exists(directory: str) -> bool:
    """Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to check/create
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False


def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get detailed information about a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return {"error": "File not found"}
        
        stat = path.stat()
        return {
            "name": path.name,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": path.suffix,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir()
        }
    except Exception as e:
        return {"error": str(e)}


def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported.
    
    Args:
        filename: Name of the file to validate
        
    Returns:
        True if file type is supported
    """
    file_ext = Path(filename).suffix.lower()
    return file_ext in SUPPORTED_FILE_TYPES


def validate_file_size(filepath: str) -> bool:
    """Validate if file size is within limits.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        True if file size is within limits
    """
    try:
        size = Path(filepath).stat().st_size
        return size <= MAX_FILE_SIZE
    except:
        return False


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove or replace invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name


def calculate_file_hash(filepath: str, algorithm: str = 'md5') -> str:
    """Calculate hash of a file.
    
    Args:
        filepath: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash as hexadecimal string
    """
    try:
        hash_obj = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        return f"Error calculating hash: {e}"


def read_json_file(filepath: str) -> Dict[str, Any]:
    """Read and parse a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def write_json_file(filepath: str, data: Dict[str, Any], indent: int = 2) -> bool:
    """Write data to a JSON file.
    
    Args:
        filepath: Path to the JSON file
        data: Data to write
        indent: JSON indentation
        
    Returns:
        True if successful
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return False


def read_csv_file(filepath: str) -> List[Dict[str, Any]]:
    """Read and parse a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of dictionaries representing CSV rows
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        return [{"error": str(e)}]


def write_csv_file(filepath: str, data: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> bool:
    """Write data to a CSV file.
    
    Args:
        filepath: Path to the CSV file
        data: List of dictionaries to write
        fieldnames: List of field names for CSV header
        
    Returns:
        True if successful
    """
    try:
        if not data:
            return False
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_timestamp(timestamp: Union[str, float, int]) -> str:
    """Format timestamp in human-readable format.
    
    Args:
        timestamp: Timestamp (ISO string, Unix timestamp, or datetime)
        
    Returns:
        Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        else:
            return str(timestamp)
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(timestamp)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction (in a real scenario, you'd use NLP libraries)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Count word frequency
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]


def create_backup(filepath: str) -> str:
    """Create a backup of a file.
    
    Args:
        filepath: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.parent / f"{path.stem}_backup_{timestamp}{path.suffix}"
        
        # Copy file
        import shutil
        shutil.copy2(filepath, backup_path)
        
        return str(backup_path)
    except Exception as e:
        print(f"Error creating backup: {e}")
        return ""


def get_system_info() -> Dict[str, Any]:
    """Get system information.
    
    Returns:
        Dictionary with system information
    """
    try:
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    except ImportError:
        return {
            "platform": "Unknown",
            "python_version": "Unknown",
            "note": "psutil not available"
        }
    except Exception as e:
        return {"error": str(e)}


def validate_input(input_data: Any, input_type: type, required: bool = True) -> Dict[str, Any]:
    """Validate input data.
    
    Args:
        input_data: Data to validate
        input_type: Expected type
        required: Whether input is required
        
    Returns:
        Validation result dictionary
    """
    result = {
        "valid": True,
        "error": None,
        "data": input_data
    }
    
    if input_data is None and required:
        result["valid"] = False
        result["error"] = "Required input is missing"
        return result
    
    if input_data is not None and not isinstance(input_data, input_type):
        result["valid"] = False
        result["error"] = f"Expected {input_type.__name__}, got {type(input_data).__name__}"
        return result
    
    return result


def create_log_entry(level: str, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a structured log entry.
    
    Args:
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        message: Log message
        context: Additional context data
        
    Returns:
        Structured log entry
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "level": level.upper(),
        "message": message,
        "context": context or {}
    }


def save_log(log_entry: Dict[str, Any], log_file: str = "agent.log") -> bool:
    """Save log entry to file.
    
    Args:
        log_entry: Log entry to save
        log_file: Log file name
        
    Returns:
        True if successful
    """
    try:
        log_path = Path(WORKSPACE_DIR) / log_file
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        return True
    except Exception as e:
        print(f"Error saving log: {e}")
        return False


def load_config_file(config_file: str = "agent_config.json") -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_file: Configuration file name
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(WORKSPACE_DIR) / config_file
    
    if not config_path.exists():
        # Return default configuration
        return {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2000,
            "tools_enabled": True,
            "memory_enabled": True,
            "logging_enabled": True
        }
    
    return read_json_file(str(config_path))


def save_config_file(config: Dict[str, Any], config_file: str = "agent_config.json") -> bool:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_file: Configuration file name
        
    Returns:
        True if successful
    """
    config_path = Path(WORKSPACE_DIR) / config_file
    return write_json_file(str(config_path), config)


def create_workspace_structure() -> bool:
    """Create the workspace directory structure.
    
    Returns:
        True if successful
    """
    directories = [
        WORKSPACE_DIR,
        f"{WORKSPACE_DIR}/data",
        f"{WORKSPACE_DIR}/logs",
        f"{WORKSPACE_DIR}/output",
        f"{WORKSPACE_DIR}/backups"
    ]
    
    success = True
    for directory in directories:
        if not ensure_directory_exists(directory):
            success = False
    
    return success


def cleanup_old_files(directory: str, max_age_days: int = 30) -> int:
    """Clean up old files from a directory.
    
    Args:
        directory: Directory to clean up
        max_age_days: Maximum age of files in days
        
    Returns:
        Number of files deleted
    """
    try:
        deleted_count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        
        for file_path in Path(directory).iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
        
        return deleted_count
    except Exception as e:
        print(f"Error cleaning up files: {e}")
        return 0
